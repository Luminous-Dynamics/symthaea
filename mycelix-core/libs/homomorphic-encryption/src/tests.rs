// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for Homomorphic Encryption
//!
//! Tests cover:
//! - Key generation
//! - Encryption/decryption correctness
//! - Homomorphic addition
//! - Homomorphic scalar multiplication
//! - Homomorphic subtraction
//! - Negative number handling
//! - Float encoding/decoding

use super::*;
use super::paillier::*;

// Use smaller key size for faster tests
const TEST_KEY_SIZE: usize = 1024;

// =============================================================================
// KEY GENERATION TESTS
// =============================================================================

#[cfg(test)]
mod key_generation_tests {
    use super::*;

    #[test]
    fn test_generate_keypair() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        // Check key sizes
        assert!(keypair.public_key.n.bits() >= TEST_KEY_SIZE / 2);
        assert_eq!(keypair.public_key.bits, TEST_KEY_SIZE);
    }

    #[test]
    fn test_keypair_components() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        // n^2 should be correct
        let expected_n_squared = &keypair.public_key.n * &keypair.public_key.n;
        assert_eq!(keypair.public_key.n_squared, expected_n_squared);

        // g = n + 1
        let expected_g = &keypair.public_key.n + BigUint::one();
        assert_eq!(keypair.public_key.g, expected_g);
    }

    #[test]
    fn test_invalid_key_size() {
        let result = PaillierKeyPair::generate(512);
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_key_sizes() {
        for &size in SUPPORTED_KEY_SIZES {
            if size <= 1024 {
                // Only test small sizes in unit tests
                let result = PaillierKeyPair::generate(size);
                assert!(result.is_ok(), "Failed for key size {}", size);
            }
        }
    }

    #[test]
    fn test_different_keypairs_are_different() {
        let kp1 = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();
        let kp2 = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        // Different key pairs should have different n
        assert_ne!(kp1.public_key.n, kp2.public_key.n);
    }
}

// =============================================================================
// ENCRYPTION/DECRYPTION TESTS
// =============================================================================

#[cfg(test)]
mod encryption_decryption_tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_positive() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        for value in [0i64, 1, 42, 100, 1000, 999999] {
            let encrypted = keypair.public_key().encrypt(value);
            let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

            assert_eq!(decrypted, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_encrypt_decrypt_negative() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        for value in [-1i64, -42, -100, -1000, -999999] {
            let encrypted = keypair.public_key().encrypt(value);
            let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

            assert_eq!(decrypted, value, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_encrypt_decrypt_zero() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let encrypted = keypair.public_key().encrypt(0);
        let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_same_value_different_ciphertexts() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        // Paillier is probabilistic - same plaintext produces different ciphertexts
        let c1 = keypair.public_key().encrypt(42);
        let c2 = keypair.public_key().encrypt(42);

        assert_ne!(c1.value, c2.value, "Ciphertexts should be different");

        // But they decrypt to the same value
        assert_eq!(keypair.decrypt_i64(&c1).unwrap(), 42);
        assert_eq!(keypair.decrypt_i64(&c2).unwrap(), 42);
    }
}

// =============================================================================
// HOMOMORPHIC ADDITION TESTS
// =============================================================================

#[cfg(test)]
mod homomorphic_addition_tests {
    use super::*;

    #[test]
    fn test_homomorphic_add() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 100i64;
        let b = 50i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_sum = enc_a.add(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_sum).unwrap();

        assert_eq!(decrypted, a + b);
    }

    #[test]
    fn test_homomorphic_add_multiple() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let values = [10i64, 20, 30, 40, 50];
        let expected_sum: i64 = values.iter().sum();

        let mut acc = keypair.public_key().encrypt(0);
        for &v in &values {
            let enc_v = keypair.public_key().encrypt(v);
            acc = acc.add(&enc_v);
        }

        let decrypted = keypair.decrypt_i64(&acc).unwrap();
        assert_eq!(decrypted, expected_sum);
    }

    #[test]
    fn test_homomorphic_add_negative() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 100i64;
        let b = -30i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_sum = enc_a.add(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_sum).unwrap();

        assert_eq!(decrypted, a + b); // 70
    }

    #[test]
    fn test_add_to_zero() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 100i64;
        let b = -100i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_sum = enc_a.add(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_sum).unwrap();

        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_add_operator_trait() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let enc_a = keypair.public_key().encrypt(25);
        let enc_b = keypair.public_key().encrypt(75);

        let enc_sum = &enc_a + &enc_b;
        let decrypted = keypair.decrypt_i64(&enc_sum).unwrap();

        assert_eq!(decrypted, 100);
    }
}

// =============================================================================
// HOMOMORPHIC SUBTRACTION TESTS
// =============================================================================

#[cfg(test)]
mod homomorphic_subtraction_tests {
    use super::*;

    #[test]
    fn test_homomorphic_sub() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 100i64;
        let b = 30i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_diff = enc_a.sub(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_diff).unwrap();

        assert_eq!(decrypted, a - b); // 70
    }

    #[test]
    fn test_homomorphic_sub_negative_result() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 30i64;
        let b = 100i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let enc_diff = enc_a.sub(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_diff).unwrap();

        assert_eq!(decrypted, a - b); // -70
    }
}

// =============================================================================
// SCALAR MULTIPLICATION TESTS
// =============================================================================

#[cfg(test)]
mod scalar_multiplication_tests {
    use super::*;

    #[test]
    fn test_scalar_mul_positive() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 10i64;
        let scalar = 5i64;

        let encrypted = keypair.public_key().encrypt(value);
        let scaled = encrypted.scalar_mul(scalar);
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, value * scalar); // 50
    }

    #[test]
    fn test_scalar_mul_negative_scalar() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 10i64;
        let scalar = -3i64;

        let encrypted = keypair.public_key().encrypt(value);
        let scaled = encrypted.scalar_mul(scalar);
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, value * scalar); // -30
    }

    #[test]
    fn test_scalar_mul_negative_value() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = -10i64;
        let scalar = 3i64;

        let encrypted = keypair.public_key().encrypt(value);
        let scaled = encrypted.scalar_mul(scalar);
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, value * scalar); // -30
    }

    #[test]
    fn test_scalar_mul_zero() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 100i64;
        let scalar = 0i64;

        let encrypted = keypair.public_key().encrypt(value);
        let scaled = encrypted.scalar_mul(scalar);
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_scalar_mul_one() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 42i64;
        let scalar = 1i64;

        let encrypted = keypair.public_key().encrypt(value);
        let scaled = encrypted.scalar_mul(scalar);
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, value);
    }

    #[test]
    fn test_mul_operator_trait() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let encrypted = keypair.public_key().encrypt(7);
        let scaled = &encrypted * 6;
        let decrypted = keypair.decrypt_i64(&scaled).unwrap();

        assert_eq!(decrypted, 42);
    }
}

// =============================================================================
// FLOAT ENCODING TESTS
// =============================================================================

#[cfg(test)]
mod float_encoding_tests {
    use super::*;

    #[test]
    fn test_encrypt_f64() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 3.14159;
        let precision = 4; // 4 decimal places

        let encrypted = keypair.public_key().encrypt_f64(value, precision);
        let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

        // Should be 31416 (3.14159 * 10000 rounded)
        assert_eq!(decrypted, 31416);
    }

    #[test]
    fn test_encrypt_f64_negative() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = -2.5;
        let precision = 2;

        let encrypted = keypair.public_key().encrypt_f64(value, precision);
        let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

        assert_eq!(decrypted, -250);
    }

    #[test]
    fn test_homomorphic_add_floats() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();
        let precision = 3;

        let a = 1.5;
        let b = 2.3;

        let enc_a = keypair.public_key().encrypt_f64(a, precision);
        let enc_b = keypair.public_key().encrypt_f64(b, precision);

        let enc_sum = enc_a.add(&enc_b);
        let decrypted = keypair.decrypt_i64(&enc_sum).unwrap();

        // 1.5 * 1000 + 2.3 * 1000 = 1500 + 2300 = 3800
        assert_eq!(decrypted, 3800);

        // Convert back to float
        let result = decrypted as f64 / 1000.0;
        assert!((result - 3.8).abs() < 0.001);
    }
}

// =============================================================================
// RERANDOMIZATION TESTS
// =============================================================================

#[cfg(test)]
mod rerandomization_tests {
    use super::*;

    #[test]
    fn test_rerandomize() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let value = 42i64;
        let encrypted = keypair.public_key().encrypt(value);
        let rerandomized = encrypted.rerandomize(keypair.public_key());

        // Ciphertext should be different
        assert_ne!(encrypted.value, rerandomized.value);

        // But decrypt to same value
        assert_eq!(keypair.decrypt_i64(&rerandomized).unwrap(), value);
    }
}

// =============================================================================
// AGGREGATION TESTS (FL USE CASE)
// =============================================================================

#[cfg(test)]
mod aggregation_tests {
    use super::*;

    #[test]
    fn test_gradient_aggregation() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();
        let precision = 4;

        // Simulate gradients from 3 clients
        let gradients = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ];

        let num_clients = gradients.len();

        // Encrypt each gradient
        let encrypted_grads: Vec<Vec<PaillierCiphertext>> = gradients
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&v| keypair.public_key().encrypt_f64(v, precision))
                    .collect()
            })
            .collect();

        // Aggregate (sum) encrypted gradients
        let mut aggregated: Vec<PaillierCiphertext> = encrypted_grads[0].clone();
        for client_grad in encrypted_grads.iter().skip(1) {
            for (i, enc_v) in client_grad.iter().enumerate() {
                aggregated[i] = aggregated[i].add(enc_v);
            }
        }

        // Decrypt and compute average
        let scale = 10i64.pow(precision);
        let decrypted: Vec<f64> = aggregated
            .iter()
            .map(|c| keypair.decrypt_i64(c).unwrap() as f64 / scale as f64 / num_clients as f64)
            .collect();

        // Expected: [0.4, 0.5, 0.6] (average of gradients)
        assert!((decrypted[0] - 0.4).abs() < 0.001);
        assert!((decrypted[1] - 0.5).abs() < 0.001);
        assert!((decrypted[2] - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_weighted_aggregation() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();
        let precision = 4;
        let scale = 10i64.pow(precision);

        // Simulate weighted aggregation
        let values = [10.0f64, 20.0, 30.0];
        let weights = [2i64, 3, 5]; // Total weight = 10

        let mut weighted_sum = keypair.public_key().encrypt(0);
        for (&v, &w) in values.iter().zip(weights.iter()) {
            let enc_v = keypair.public_key().encrypt_f64(v, precision);
            let weighted = enc_v.scalar_mul(w);
            weighted_sum = weighted_sum.add(&weighted);
        }

        let sum = keypair.decrypt_i64(&weighted_sum).unwrap() as f64 / scale as f64;
        // 10*2 + 20*3 + 30*5 = 20 + 60 + 150 = 230
        assert!((sum - 230.0).abs() < 0.01);

        // Average: 230 / 10 = 23
        let average = sum / weights.iter().sum::<i64>() as f64;
        assert!((average - 23.0).abs() < 0.01);
    }
}

// =============================================================================
// EDGE CASES AND PROPERTY TESTS
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_large_values() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let large_value = 1_000_000_000i64;
        let encrypted = keypair.public_key().encrypt(large_value);
        let decrypted = keypair.decrypt_i64(&encrypted).unwrap();

        assert_eq!(decrypted, large_value);
    }

    #[test]
    fn test_many_additions() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let mut acc = keypair.public_key().encrypt(0);
        let count = 100;

        for _ in 0..count {
            let enc_one = keypair.public_key().encrypt(1);
            acc = acc.add(&enc_one);
        }

        let decrypted = keypair.decrypt_i64(&acc).unwrap();
        assert_eq!(decrypted, count);
    }

    #[test]
    fn test_commutativity() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 100i64;
        let b = 200i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        let ab = enc_a.add(&enc_b);
        let ba = enc_b.add(&enc_a);

        assert_eq!(
            keypair.decrypt_i64(&ab).unwrap(),
            keypair.decrypt_i64(&ba).unwrap()
        );
    }

    #[test]
    fn test_associativity() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 10i64;
        let b = 20i64;
        let c = 30i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);
        let enc_c = keypair.public_key().encrypt(c);

        let ab_c = enc_a.add(&enc_b).add(&enc_c);
        let a_bc = enc_a.add(&enc_b.add(&enc_c));

        assert_eq!(
            keypair.decrypt_i64(&ab_c).unwrap(),
            keypair.decrypt_i64(&a_bc).unwrap()
        );
    }

    #[test]
    fn test_distributivity() {
        let keypair = PaillierKeyPair::generate(TEST_KEY_SIZE).unwrap();

        let a = 10i64;
        let b = 20i64;
        let k = 3i64;

        let enc_a = keypair.public_key().encrypt(a);
        let enc_b = keypair.public_key().encrypt(b);

        // k(a + b)
        let sum_then_scale = enc_a.add(&enc_b).scalar_mul(k);

        // ka + kb
        let scale_then_sum = enc_a.scalar_mul(k).add(&enc_b.scalar_mul(k));

        assert_eq!(
            keypair.decrypt_i64(&sum_then_scale).unwrap(),
            keypair.decrypt_i64(&scale_then_sum).unwrap()
        );
    }
}
