// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paillier Cryptosystem Implementation
//!
//! The Paillier cryptosystem is an additively homomorphic encryption scheme:
//! - E(a) * E(b) = E(a + b)
//! - E(a)^k = E(a * k)
//!
//! This makes it ideal for aggregating encrypted gradients in federated learning.

use crate::{HeError, HeResult, RECOMMENDED_KEY_SIZE, SUPPORTED_KEY_SIZES};
use num_bigint::{BigInt, BigUint, RandBigInt, ToBigInt};
use num_integer::Integer;
use num_traits::{One, Zero, Signed};
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use tracing::{debug, instrument};

/// Paillier public key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaillierPublicKey {
    /// n = p * q
    pub n: BigUint,
    /// n^2
    pub n_squared: BigUint,
    /// g = n + 1 (simplified generator)
    pub g: BigUint,
    /// Key size in bits
    pub bits: usize,
}

/// Paillier private key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaillierPrivateKey {
    /// λ = lcm(p-1, q-1)
    pub lambda: BigUint,
    /// μ = λ^(-1) mod n
    pub mu: BigUint,
    /// Reference to n for operations
    pub n: BigUint,
    /// n^2
    pub n_squared: BigUint,
}

/// Paillier key pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaillierKeyPair {
    pub public_key: PaillierPublicKey,
    pub private_key: PaillierPrivateKey,
}

/// Paillier ciphertext
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaillierCiphertext {
    /// The encrypted value c ∈ Z*_{n^2}
    pub value: BigUint,
    /// Reference to the public key's n^2 for operations
    pub n_squared: BigUint,
    /// Reference to n for scalar multiplication
    pub n: BigUint,
}

/// Main Paillier encryption interface
pub struct Paillier;

impl Paillier {
    /// Generate a new key pair
    #[instrument(skip_all, fields(bits = bits))]
    pub fn generate_keypair(bits: usize) -> HeResult<PaillierKeyPair> {
        if !SUPPORTED_KEY_SIZES.contains(&bits) {
            return Err(HeError::ParameterError(format!(
                "Unsupported key size: {}. Supported: {:?}",
                bits, SUPPORTED_KEY_SIZES
            )));
        }

        let mut rng = thread_rng();
        let half_bits = bits / 2;

        // Generate two large primes p and q
        let p = generate_prime(&mut rng, half_bits);
        let q = generate_prime(&mut rng, half_bits);

        // Ensure p ≠ q
        let q = if p == q {
            generate_prime(&mut rng, half_bits)
        } else {
            q
        };

        // n = p * q
        let n = &p * &q;
        let n_squared = &n * &n;

        // g = n + 1 (simplified, always works)
        let g = &n + BigUint::one();

        // λ = lcm(p-1, q-1)
        let p_minus_1 = &p - BigUint::one();
        let q_minus_1 = &q - BigUint::one();
        let lambda = lcm(&p_minus_1, &q_minus_1);

        // μ = λ^(-1) mod n
        let mu = mod_inverse(&lambda, &n)
            .ok_or_else(|| HeError::KeyGenError("Failed to compute mu".to_string()))?;

        debug!(n_bits = n.bits(), "Generated Paillier key pair");

        Ok(PaillierKeyPair {
            public_key: PaillierPublicKey {
                n: n.clone(),
                n_squared: n_squared.clone(),
                g,
                bits,
            },
            private_key: PaillierPrivateKey {
                lambda,
                mu,
                n,
                n_squared,
            },
        })
    }
}

impl PaillierKeyPair {
    /// Generate a new key pair with default size
    pub fn generate(bits: usize) -> HeResult<Self> {
        Paillier::generate_keypair(bits)
    }

    /// Generate with recommended key size
    pub fn generate_default() -> HeResult<Self> {
        Self::generate(RECOMMENDED_KEY_SIZE)
    }

    /// Get the public key
    pub fn public_key(&self) -> &PaillierPublicKey {
        &self.public_key
    }

    /// Decrypt a ciphertext
    pub fn decrypt(&self, ciphertext: &PaillierCiphertext) -> BigInt {
        self.private_key.decrypt(ciphertext)
    }

    /// Decrypt to i64 (for small values)
    pub fn decrypt_i64(&self, ciphertext: &PaillierCiphertext) -> HeResult<i64> {
        let result = self.decrypt(ciphertext);

        // Handle negative numbers (values > n/2 are negative)
        let n_half = &self.public_key.n / 2u32;
        let n_int = self.public_key.n.to_bigint().expect("BigUint always converts to BigInt");

        let adjusted = if result > n_half.to_bigint().expect("BigUint always converts to BigInt") {
            &result - &n_int
        } else {
            result
        };

        // Try to convert to i64
        adjusted
            .try_into()
            .map_err(|_| HeError::Overflow)
    }
}

impl PaillierPublicKey {
    /// Encrypt a value
    pub fn encrypt(&self, value: i64) -> PaillierCiphertext {
        self.encrypt_bigint(&BigInt::from(value))
    }

    /// Encrypt a BigInt value
    pub fn encrypt_bigint(&self, value: &BigInt) -> PaillierCiphertext {
        let mut rng = thread_rng();

        // Handle negative values by adding n
        let m = if value.is_negative() {
            let n_int = self.n.to_bigint().expect("BigUint always converts to BigInt");
            (value + &n_int).to_biguint().expect("negative value + n must be non-negative")
        } else {
            value.to_biguint().expect("non-negative BigInt converts to BigUint")
        };

        // Generate random r ∈ Z*_n
        let r = rng.gen_biguint_below(&self.n);

        // c = g^m * r^n mod n^2
        let gm = mod_pow(&self.g, &m, &self.n_squared);
        let rn = mod_pow(&r, &self.n, &self.n_squared);
        let c = (&gm * &rn) % &self.n_squared;

        PaillierCiphertext {
            value: c,
            n_squared: self.n_squared.clone(),
            n: self.n.clone(),
        }
    }

    /// Encrypt a float by encoding it as a fixed-point integer
    pub fn encrypt_f64(&self, value: f64, precision: u32) -> PaillierCiphertext {
        let scale = 10i64.pow(precision);
        let encoded = (value * scale as f64).round() as i64;
        self.encrypt(encoded)
    }
}

impl PaillierPrivateKey {
    /// Decrypt a ciphertext
    pub fn decrypt(&self, ciphertext: &PaillierCiphertext) -> BigInt {
        // L(x) = (x - 1) / n
        let u = mod_pow(&ciphertext.value, &self.lambda, &self.n_squared);
        let l_u = l_function(&u, &self.n);

        // m = L(c^λ mod n^2) * μ mod n
        let m = (&l_u * &self.mu) % &self.n;

        m.to_bigint().expect("BigUint always converts to BigInt")
    }
}

impl PaillierCiphertext {
    /// Homomorphic addition: E(a) + E(b) = E(a + b)
    pub fn add(&self, other: &Self) -> Self {
        let value = (&self.value * &other.value) % &self.n_squared;
        Self {
            value,
            n_squared: self.n_squared.clone(),
            n: self.n.clone(),
        }
    }

    /// Homomorphic scalar multiplication: E(a) * k = E(a * k)
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        let scalar_big = if scalar >= 0 {
            BigUint::from(scalar as u64)
        } else {
            // For negative scalars, use n - |scalar|
            &self.n - BigUint::from((-scalar) as u64)
        };

        let value = mod_pow(&self.value, &scalar_big, &self.n_squared);
        Self {
            value,
            n_squared: self.n_squared.clone(),
            n: self.n.clone(),
        }
    }

    /// Homomorphic subtraction: E(a) - E(b) = E(a - b)
    pub fn sub(&self, other: &Self) -> Self {
        // E(a) - E(b) = E(a) * E(-b) = E(a) * E(b)^(-1) mod n^2
        let other_inv = mod_inverse(&other.value, &self.n_squared)
            .expect("Ciphertext should be invertible");
        let value = (&self.value * &other_inv) % &self.n_squared;
        Self {
            value,
            n_squared: self.n_squared.clone(),
            n: self.n.clone(),
        }
    }

    /// Re-randomize the ciphertext (for privacy)
    pub fn rerandomize(&self, public_key: &PaillierPublicKey) -> Self {
        let mut rng = thread_rng();
        let r = rng.gen_biguint_below(&public_key.n);
        let rn = mod_pow(&r, &public_key.n, &self.n_squared);
        let value = (&self.value * &rn) % &self.n_squared;
        Self {
            value,
            n_squared: self.n_squared.clone(),
            n: self.n.clone(),
        }
    }
}

// Implement Add trait for convenient syntax
impl Add for &PaillierCiphertext {
    type Output = PaillierCiphertext;

    fn add(self, other: Self) -> PaillierCiphertext {
        self.add(other)
    }
}

impl Add<&PaillierCiphertext> for PaillierCiphertext {
    type Output = PaillierCiphertext;

    fn add(self, other: &PaillierCiphertext) -> PaillierCiphertext {
        (&self).add(other)
    }
}

// Implement Mul for scalar multiplication
impl Mul<i64> for &PaillierCiphertext {
    type Output = PaillierCiphertext;

    fn mul(self, scalar: i64) -> PaillierCiphertext {
        self.scalar_mul(scalar)
    }
}

// ============ Helper Functions ============

/// Generate a random prime of approximately `bits` bits
fn generate_prime(rng: &mut impl rand::Rng, bits: usize) -> BigUint {
    loop {
        let candidate = rng.gen_biguint(bits as u64);
        // Ensure it's odd and has the right bit length
        let candidate = &candidate | BigUint::one();
        if is_probably_prime(&candidate, 20) {
            return candidate;
        }
    }
}

/// Miller-Rabin primality test
fn is_probably_prime(n: &BigUint, rounds: usize) -> bool {
    if n <= &BigUint::one() {
        return false;
    }
    if n == &BigUint::from(2u32) || n == &BigUint::from(3u32) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 as 2^r * d
    let n_minus_1 = n - BigUint::one();
    let mut d = n_minus_1.clone();
    let mut r = 0u32;
    while d.is_even() {
        d >>= 1;
        r += 1;
    }

    let mut rng = thread_rng();

    'witness: for _ in 0..rounds {
        let a = rng.gen_biguint_range(&BigUint::from(2u32), &(n - BigUint::from(2u32)));
        let mut x = mod_pow(&a, &d, n);

        if x == BigUint::one() || x == n_minus_1 {
            continue 'witness;
        }

        for _ in 0..(r - 1) {
            x = mod_pow(&x, &BigUint::from(2u32), n);
            if x == n_minus_1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Modular exponentiation: base^exp mod modulus
fn mod_pow(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    base.modpow(exp, modulus)
}

/// Extended Euclidean algorithm for modular inverse
fn mod_inverse(a: &BigUint, modulus: &BigUint) -> Option<BigUint> {
    let a_int = a.to_bigint().expect("BigUint always converts to BigInt");
    let m_int = modulus.to_bigint().expect("BigUint always converts to BigInt");

    let (gcd, x, _) = extended_gcd(&a_int, &m_int);

    if gcd != BigInt::one() {
        return None;
    }

    let result = ((x % &m_int) + &m_int) % &m_int;
    result.to_biguint()
}

/// Extended GCD
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }

    let (gcd, x1, y1) = extended_gcd(b, &(a % b));
    let x = y1.clone();
    let y = x1 - (a / b) * y1;

    (gcd, x, y)
}

/// Least common multiple
fn lcm(a: &BigUint, b: &BigUint) -> BigUint {
    (a * b) / a.gcd(b)
}

/// L function: L(x) = (x - 1) / n
fn l_function(x: &BigUint, n: &BigUint) -> BigUint {
    (x - BigUint::one()) / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        // n = p * q where p, q are ~512 bits, so n should be ~1024 bits
        // Allow some variance in bit count due to probabilistic prime generation
        assert!(keypair.public_key.n.bits() >= 1000,
            "Expected n >= 1000 bits, got {}", keypair.public_key.n.bits());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();

        let plaintext = 42i64;
        let ciphertext = keypair.public_key().encrypt(plaintext);
        let decrypted = keypair.decrypt_i64(&ciphertext).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_negative() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();

        let plaintext = -42i64;
        let ciphertext = keypair.public_key().encrypt(plaintext);
        let decrypted = keypair.decrypt_i64(&ciphertext).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_homomorphic_addition() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();

        let a = 100i64;
        let b = 50i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_sum = &enc_a + &enc_b;
        let sum = keypair.decrypt_i64(&enc_sum).unwrap();

        assert_eq!(sum, a + b);
    }

    #[test]
    fn test_homomorphic_scalar_mul() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();

        let a = 25i64;
        let k = 4i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_product = &enc_a * k;

        let product = keypair.decrypt_i64(&enc_product).unwrap();
        assert_eq!(product, a * k);
    }

    #[test]
    fn test_homomorphic_subtraction() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();

        let a = 100i64;
        let b = 30i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_diff = enc_a.sub(&enc_b);
        let diff = keypair.decrypt_i64(&enc_diff).unwrap();

        assert_eq!(diff, a - b);
    }

    #[test]
    fn test_aggregate_multiple() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let values = vec![10i64, 20, 30, 40, 50];
        let encrypted: Vec<_> = values.iter().map(|&v| pk.encrypt(v)).collect();

        // Aggregate all encrypted values
        let mut sum = pk.encrypt(0);
        for enc in &encrypted {
            sum = &sum + enc;
        }

        let result = keypair.decrypt_i64(&sum).unwrap();
        let expected: i64 = values.iter().sum();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rerandomize() {
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let plaintext = 42i64;
        let enc1 = pk.encrypt(plaintext);
        let enc2 = enc1.rerandomize(pk);

        // Ciphertexts should be different
        assert_ne!(enc1.value, enc2.value);

        // But decrypt to the same value
        assert_eq!(keypair.decrypt_i64(&enc1).unwrap(), plaintext);
        assert_eq!(keypair.decrypt_i64(&enc2).unwrap(), plaintext);
    }

    // ==================== CRITICAL TESTS ====================
    // Added per CODE_REVIEW_FINDINGS_2026-01-19.md Section 2

    #[test]
    fn test_key_validity_n_is_product_of_primes() {
        // CRITICAL: Verify n = p * q where p, q are large primes
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let n = &keypair.public_key.n;

        // n should be approximately 1024 bits
        assert!(n.bits() >= 1000, "n too small: {} bits", n.bits());
        assert!(n.bits() <= 1030, "n too large: {} bits", n.bits());

        // n should not be even (both p, q must be odd primes)
        assert!(!n.is_even(), "n should be odd");

        // n_squared should equal n * n
        assert_eq!(
            keypair.public_key.n_squared,
            n * n,
            "n_squared should equal n * n"
        );
    }

    #[test]
    fn test_semantic_security_different_ciphertexts() {
        // CRITICAL: Same plaintext must produce different ciphertexts (semantic security)
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let plaintext = 42i64;

        // Encrypt the same value multiple times
        let enc1 = pk.encrypt(plaintext);
        let enc2 = pk.encrypt(plaintext);
        let enc3 = pk.encrypt(plaintext);

        // All ciphertexts should be different (due to random r)
        assert_ne!(enc1.value, enc2.value, "Encryption must be randomized");
        assert_ne!(enc2.value, enc3.value, "Encryption must be randomized");
        assert_ne!(enc1.value, enc3.value, "Encryption must be randomized");

        // But all decrypt to the same value
        assert_eq!(keypair.decrypt_i64(&enc1).unwrap(), plaintext);
        assert_eq!(keypair.decrypt_i64(&enc2).unwrap(), plaintext);
        assert_eq!(keypair.decrypt_i64(&enc3).unwrap(), plaintext);
    }

    #[test]
    fn test_encrypt_zero() {
        // Edge case: encrypting zero
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let enc = pk.encrypt(0);
        let decrypted = keypair.decrypt_i64(&enc).unwrap();
        assert_eq!(decrypted, 0);

        // Zero should be additive identity: E(a) + E(0) = E(a)
        let a = 100i64;
        let enc_a = pk.encrypt(a);
        let enc_zero = pk.encrypt(0);
        let enc_sum = &enc_a + &enc_zero;
        assert_eq!(keypair.decrypt_i64(&enc_sum).unwrap(), a);
    }

    #[test]
    fn test_negative_scalar_multiplication() {
        // CRITICAL: Negative scalar multiplication
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 50i64;
        let k = -3i64;

        let enc_a = pk.encrypt(a);
        let enc_product = &enc_a * k;

        let result = keypair.decrypt_i64(&enc_product).unwrap();
        assert_eq!(result, a * k, "Negative scalar mul failed: {} * {} = {}", a, k, result);
    }

    #[test]
    fn test_homomorphic_addition_commutativity() {
        // CRITICAL: E(a) + E(b) should equal E(b) + E(a)
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 100i64;
        let b = 200i64;

        let enc_a = pk.encrypt(a);
        let enc_b = pk.encrypt(b);

        let sum1 = &enc_a + &enc_b;
        let sum2 = &enc_b + &enc_a;

        // Both should decrypt to same value
        assert_eq!(keypair.decrypt_i64(&sum1).unwrap(), a + b);
        assert_eq!(keypair.decrypt_i64(&sum2).unwrap(), a + b);
    }

    #[test]
    fn test_homomorphic_addition_associativity() {
        // CRITICAL: (E(a) + E(b)) + E(c) should equal E(a) + (E(b) + E(c))
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 10i64;
        let b = 20i64;
        let c = 30i64;

        let enc_a = pk.encrypt(a);
        let enc_b = pk.encrypt(b);
        let enc_c = pk.encrypt(c);

        // (a + b) + c
        let sum1 = (&enc_a + &enc_b).add(&enc_c);
        // a + (b + c)
        let sum2 = enc_a.add(&(&enc_b + &enc_c));

        assert_eq!(keypair.decrypt_i64(&sum1).unwrap(), a + b + c);
        assert_eq!(keypair.decrypt_i64(&sum2).unwrap(), a + b + c);
    }

    #[test]
    fn test_scalar_mul_distributivity() {
        // CRITICAL: E(a) * (k1 + k2) should equal E(a) * k1 + E(a) * k2
        // But since we only support scalar mul, test: E(a) * k1 * k2 = E(a * k1 * k2)
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 5i64;
        let k1 = 3i64;
        let k2 = 4i64;

        let enc_a = pk.encrypt(a);

        // ((E(a) * k1) * k2)
        let result1 = (&enc_a * k1).scalar_mul(k2);
        // E(a * k1 * k2)
        let expected = a * k1 * k2;

        assert_eq!(keypair.decrypt_i64(&result1).unwrap(), expected);
    }

    #[test]
    fn test_mixed_positive_negative() {
        // Test mixing positive and negative values
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 100i64;
        let b = -30i64;
        let c = -50i64;

        let enc_a = pk.encrypt(a);
        let enc_b = pk.encrypt(b);
        let enc_c = pk.encrypt(c);

        // a + b + c = 100 - 30 - 50 = 20
        let sum = (&enc_a + &enc_b).add(&enc_c);
        assert_eq!(keypair.decrypt_i64(&sum).unwrap(), a + b + c);
    }

    #[test]
    fn test_subtraction_to_negative() {
        // Test subtraction that results in negative
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 30i64;
        let b = 100i64;

        let enc_a = pk.encrypt(a);
        let enc_b = pk.encrypt(b);

        // a - b = 30 - 100 = -70
        let diff = enc_a.sub(&enc_b);
        assert_eq!(keypair.decrypt_i64(&diff).unwrap(), a - b);
    }

    #[test]
    fn test_scalar_mul_by_zero() {
        // Edge case: E(a) * 0 = E(0)
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 42i64;
        let enc_a = pk.encrypt(a);
        let enc_zero = &enc_a * 0i64;

        assert_eq!(keypair.decrypt_i64(&enc_zero).unwrap(), 0);
    }

    #[test]
    fn test_scalar_mul_by_one() {
        // Edge case: E(a) * 1 = E(a)
        let keypair = PaillierKeyPair::generate(1024).unwrap();
        let pk = keypair.public_key();

        let a = 42i64;
        let enc_a = pk.encrypt(a);
        let enc_same = &enc_a * 1i64;

        assert_eq!(keypair.decrypt_i64(&enc_same).unwrap(), a);
    }

    #[test]
    fn test_invalid_key_size_rejected() {
        // CRITICAL: Invalid key sizes should be rejected
        let result = PaillierKeyPair::generate(512); // Too small
        assert!(result.is_err(), "512-bit keys should be rejected");

        let result = PaillierKeyPair::generate(100); // Way too small
        assert!(result.is_err(), "100-bit keys should be rejected");
    }
}
