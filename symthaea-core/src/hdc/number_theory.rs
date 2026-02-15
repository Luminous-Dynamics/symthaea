#![allow(dead_code)]

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{seed_from_name, PrimitiveSystem};

/// Prime factorization of a number with HDC encoding
#[derive(Debug, Clone)]
pub struct PrimeFactorization {
    pub value: u64,
    pub factors: Vec<(u64, u32)>,
    pub encoding: BinaryHV,
    pub phi: f64,
}

/// Number theory engine for HDC-based arithmetic
pub struct NumberTheoryEngine {
    multiplication: BinaryHV,
    addition: BinaryHV,
    zero: BinaryHV,
    successor: BinaryHV,
}

impl NumberTheoryEngine {
    pub fn new() -> Self {
        let prims = PrimitiveSystem::global();
        Self {
            multiplication: prims
                .get("MULTIPLICATION")
                .expect("MULTIPLICATION primitive must exist")
                .encoding,
            addition: prims
                .get("ADDITION")
                .expect("ADDITION primitive must exist")
                .encoding,
            zero: prims
                .get("ZERO")
                .expect("ZERO primitive must exist")
                .encoding,
            successor: prims
                .get("SUCCESSOR")
                .expect("SUCCESSOR primitive must exist")
                .encoding,
        }
    }

    /// Trial division factorization with HDC encoding
    pub fn factorize(&self, n: u64) -> PrimeFactorization {
        if n == 0 {
            return PrimeFactorization {
                value: 0,
                factors: vec![],
                encoding: self.zero,
                phi: 0.0,
            };
        }

        if n == 1 {
            return PrimeFactorization {
                value: 1,
                factors: vec![],
                encoding: self.successor.bind(&self.zero),
                phi: 1.0,
            };
        }

        let mut factors = Vec::new();
        let mut remaining = n;
        let mut divisor = 2;

        while divisor * divisor <= remaining {
            let mut count = 0;
            while remaining.is_multiple_of(divisor) {
                count += 1;
                remaining /= divisor;
            }
            if count > 0 {
                factors.push((divisor, count));
            }
            divisor += if divisor == 2 { 1 } else { 2 };
        }

        if remaining > 1 {
            factors.push((remaining, 1));
        }

        // Encode as bundle of MULTIPLICATION.bind(prime_enc).bind(exp_enc)
        let mut factor_encodings = Vec::new();
        for (prime, exponent) in &factors {
            let prime_enc = BinaryHV::random(seed_from_name(&format!("prime_{prime}")));
            let exp_enc = BinaryHV::random(seed_from_name(&format!("exp_{exponent}")));
            let factor_enc = self.multiplication.bind(&prime_enc).bind(&exp_enc);
            factor_encodings.push(factor_enc);
        }

        let encoding = if factor_encodings.is_empty() {
            self.successor.bind(&self.zero)
        } else {
            BinaryHV::bundle(&factor_encodings)
        };

        // Calculate Euler's totient function: φ(n) = n * ∏(1 - 1/p)
        let mut phi = n as f64;
        for (prime, _) in &factors {
            phi *= 1.0 - 1.0 / (*prime as f64);
        }

        PrimeFactorization {
            value: n,
            factors,
            encoding,
            phi,
        }
    }

    /// Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd
    pub fn extended_gcd(&self, a: i64, b: i64) -> (i64, i64, i64) {
        if b == 0 {
            return (a.abs(), if a >= 0 { 1 } else { -1 }, 0);
        }

        let mut old_r = a;
        let mut r = b;
        let mut old_s = 1i64;
        let mut s = 0i64;
        let mut old_t = 0i64;
        let mut t = 1i64;

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;

            let temp_t = t;
            t = old_t - quotient * t;
            old_t = temp_t;
        }

        (old_r.abs(), old_s, old_t)
    }

    /// Greatest common divisor using Euclidean algorithm
    pub fn gcd(&self, a: u64, b: u64) -> u64 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    /// Least common multiple
    pub fn lcm(&self, a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            return 0;
        }
        (a / self.gcd(a, b)) * b
    }

    /// Sieve of Eratosthenes
    pub fn primes_up_to(&self, n: u64) -> Vec<(u64, BinaryHV)> {
        if n < 2 {
            return vec![];
        }

        let n_size = n as usize + 1;
        let mut is_prime = vec![true; n_size];
        is_prime[0] = false;
        is_prime[1] = false;

        let limit = (n as f64).sqrt() as usize;
        for i in 2..=limit {
            if is_prime[i] {
                let mut j = i * i;
                while j < n_size {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }

        is_prime
            .iter()
            .enumerate()
            .filter(|(_, &prime)| prime)
            .map(|(i, _)| {
                let p = i as u64;
                let encoding = BinaryHV::random(seed_from_name(&format!("prime_{p}")));
                (p, encoding)
            })
            .collect()
    }

    /// Trial division primality test
    pub fn is_prime(&self, n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let limit = (n as f64).sqrt() as u64;
        let mut divisor = 3;
        while divisor <= limit {
            if n.is_multiple_of(divisor) {
                return false;
            }
            divisor += 2;
        }
        true
    }

    /// Fermat primality test: for each witness a, check a^(n-1) mod n == 1
    pub fn fermat_test(&self, n: u64, witnesses: &[u64]) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let ring = ModularRing::new(n);
        for &a in witnesses {
            if a >= n {
                continue;
            }
            if self.gcd(a, n) != 1 {
                return false;
            }
            if ring.power(a, n - 1) != 1 {
                return false;
            }
        }
        true
    }
}

impl Default for NumberTheoryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Modular arithmetic ring Z/nZ
pub struct ModularRing {
    pub modulus: u64,
    pub modulus_encoding: BinaryHV,
}

impl ModularRing {
    pub fn new(modulus: u64) -> Self {
        let modulus_encoding = BinaryHV::random(seed_from_name(&format!("mod_{modulus}")));
        Self {
            modulus,
            modulus_encoding,
        }
    }

    /// Modular addition
    pub fn add(&self, a: u64, b: u64) -> u64 {
        let a = a % self.modulus;
        let b = b % self.modulus;
        (a + b) % self.modulus
    }

    /// Modular multiplication
    pub fn multiply(&self, a: u64, b: u64) -> u64 {
        let a = a % self.modulus;
        let b = b % self.modulus;
        ((a as u128 * b as u128) % self.modulus as u128) as u64
    }

    /// Modular exponentiation by squaring
    pub fn power(&self, base: u64, exp: u64) -> u64 {
        if exp == 0 {
            return 1;
        }

        let mut result = 1u64;
        let mut base = base % self.modulus;
        let mut exp = exp;

        while exp > 0 {
            if exp % 2 == 1 {
                result = self.multiply(result, base);
            }
            base = self.multiply(base, base);
            exp /= 2;
        }

        result
    }

    /// Modular multiplicative inverse via extended Euclidean algorithm
    pub fn inverse(&self, a: u64) -> Option<u64> {
        let engine = NumberTheoryEngine::new();
        let (gcd, x, _) = engine.extended_gcd(a as i64, self.modulus as i64);

        if gcd != 1 {
            return None;
        }

        let inv = ((x % self.modulus as i64 + self.modulus as i64) % self.modulus as i64) as u64;
        Some(inv)
    }

    /// Check if element is a unit (has multiplicative inverse)
    pub fn is_unit(&self, a: u64) -> bool {
        let engine = NumberTheoryEngine::new();
        engine.gcd(a, self.modulus) == 1
    }

    /// Euler's totient function: count of units in Z/nZ
    pub fn euler_totient(&self) -> u64 {
        (1..self.modulus).filter(|&a| self.is_unit(a)).count() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_60() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(60);
        assert_eq!(f.factors, vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_factorize_1() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(1);
        assert!(f.factors.is_empty());
    }

    #[test]
    fn test_factorize_prime() {
        let engine = NumberTheoryEngine::new();
        let f = engine.factorize(17);
        assert_eq!(f.factors, vec![(17, 1)]);
    }

    #[test]
    fn test_extended_gcd() {
        let engine = NumberTheoryEngine::new();
        let (g, x, y) = engine.extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_modular_inverse() {
        let ring = ModularRing::new(7);
        let inv = ring.inverse(3).unwrap();
        assert_eq!(inv, 5); // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(ring.multiply(3, inv), 1);
    }

    #[test]
    fn test_no_inverse() {
        let ring = ModularRing::new(6);
        assert!(ring.inverse(2).is_none()); // gcd(2, 6) = 2 != 1
    }

    #[test]
    fn test_modular_power() {
        let ring = ModularRing::new(13);
        assert_eq!(ring.power(2, 10), 10); // 2^10 = 1024, 1024 mod 13 = 10
    }

    #[test]
    fn test_euler_totient() {
        let ring = ModularRing::new(12);
        assert_eq!(ring.euler_totient(), 4); // {1, 5, 7, 11}
    }

    #[test]
    fn test_primes_up_to() {
        let engine = NumberTheoryEngine::new();
        let primes = engine.primes_up_to(20);
        let prime_values: Vec<u64> = primes.iter().map(|(p, _)| *p).collect();
        assert_eq!(prime_values, vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_is_prime() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.is_prime(17));
        assert!(!engine.is_prime(15));
        assert!(!engine.is_prime(1));
        assert!(engine.is_prime(2));
    }

    #[test]
    fn test_fermat_test() {
        let engine = NumberTheoryEngine::new();
        assert!(engine.fermat_test(17, &[2, 3, 5]));
        assert!(!engine.fermat_test(15, &[2]));
    }

    #[test]
    fn test_lcm() {
        let engine = NumberTheoryEngine::new();
        assert_eq!(engine.lcm(12, 18), 36);
    }
}
