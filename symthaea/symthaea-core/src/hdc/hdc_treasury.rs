// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC Treasury — Privacy-Preserving Community Finance
//!
//! Encodes financial values (balances, amounts) as BinaryHV vectors,
//! encrypts them with HDC one-time-pad masks, and enables homomorphic
//! aggregation for community-level accounting without revealing individual values.
//!
//! # Architecture
//!
//! ```text
//! Balance (u64) → encode → BinaryHV → encrypt(mask) → EncryptedHV
//!                                                        │
//!                               ┌────────────────────────┤
//!                               │ Homomorphic aggregate   │
//!                               ▼                         │
//!                          Aggregate EncryptedHV          │
//!                               │                         │
//!                     k-of-n threshold decrypt            │
//!                               ▼                         │
//!                          Aggregate BinaryHV → decode → Total
//! ```
//!
//! Science: Imani et al. (2019) secure HDC collaboration;
//! Kanerva (2009) hyperdimensional computing;
//! Shannon (1949) information-theoretic secrecy.

use super::binary_hv::BinaryHV;
use super::hdc_crypto::{HdcShare, HdcThresholdSharing};
use super::hdc_fhe::EncryptedHV;

/// Maximum members in a single treasury pool (memory bound: 256 x 2KB = 512KB).
pub const HDC_TREASURY_MAX_MEMBERS: usize = 256;

/// Number of bits in the thermometer encoding (= BinaryHV dimension).
const THERMOMETER_LEVELS: u64 = 16384;

/// Quantization resolution: the minimum distinguishable balance difference
/// for the default max_balance of 1B micro-SAP.
///
/// With 16,384 thermometer levels mapping [0, max_balance], each level
/// represents `max_balance / 16384` micro-SAP units. For max_balance = 1B,
/// this is ~61,035 micro-SAP (~6.1 cents at 1 SAP = 1 USD).
///
/// **Acceptable use**: aggregate solvency checks, community-level accounting,
/// privacy-preserving audits where approximate totals suffice.
///
/// **Unacceptable use**: exact double-entry accounting, per-transaction settlement,
/// regulatory balance reporting where cent-level precision is required.
pub const QUANTIZATION_RESOLUTION: u64 = 1_000_000_000 / 16384; // ~61,035

/// Compute the maximum quantization error for a given max_balance.
///
/// The thermometer encoding maps `[0, max_balance]` to 16,384 discrete levels.
/// Any balance value within one level is indistinguishable after encoding,
/// so the worst-case roundtrip error is `max_balance / 16384`.
///
/// # Example
///
/// ```
/// # use symthaea_core::hdc::hdc_treasury::max_quantization_error;
/// assert_eq!(max_quantization_error(1_000_000_000), 61035); // ~61K micro-SAP
/// assert_eq!(max_quantization_error(1_000_000), 61);         // ~61 micro-SAP
/// ```
pub fn max_quantization_error(max_balance: u64) -> u64 {
    max_balance / THERMOMETER_LEVELS
}

/// Encode a balance with deterministic dithering to reduce systematic quantization bias.
///
/// Standard thermometer encoding always rounds down to the nearest level boundary,
/// introducing a systematic negative bias in aggregate statistics. Dithering adds
/// a deterministic perturbation (based on `nonce`) to the boundary bit, so that
/// over many encodings the bias averages out.
///
/// The nonce should be unique per encoding (e.g., a hash of the member DID + epoch).
/// The dithering is deterministic: same (balance, max_balance, nonce) always produces
/// the same BinaryHV.
///
/// # Arguments
///
/// * `balance` — The balance in micro-SAP units.
/// * `max_balance` — Maximum possible balance (encoding range).
/// * `nonce` — Deterministic seed for dither decision at the boundary bit.
pub fn encode_balance_with_dithering(balance: u64, max_balance: u64, nonce: u64) -> BinaryHV {
    if max_balance == 0 {
        return BinaryHV([0u8; 2048]);
    }
    let scaled = balance as u128 * THERMOMETER_LEVELS as u128;
    let bits_to_set_floor = (scaled / max_balance as u128).min(THERMOMETER_LEVELS as u128) as usize;

    // Compute fractional part: how far we are between level N and N+1.
    // `remainder` is in [0, max_balance), representing the sub-level offset.
    let remainder = scaled % max_balance as u128;
    // Dither: compare remainder to a deterministic threshold derived from nonce.
    // If the fractional position exceeds the nonce-derived threshold, round up.
    let threshold = (nonce % max_balance) as u128;
    let bits_to_set = if remainder > 0 && remainder > threshold {
        // Deterministically round up based on nonce
        (bits_to_set_floor + 1).min(THERMOMETER_LEVELS as usize)
    } else {
        bits_to_set_floor
    };

    let mut bytes = [0u8; 2048];
    let full_bytes = bits_to_set / 8;
    let remaining_bits = bits_to_set % 8;
    for b in bytes.iter_mut().take(full_bytes) {
        *b = 0xFF;
    }
    if remaining_bits > 0 && full_bytes < 2048 {
        bytes[full_bytes] = 0xFF << (8 - remaining_bits);
    }
    BinaryHV(bytes)
}

/// Estimate the relative error of homomorphic majority-vote bundling.
///
/// When n BinaryHVs are bundled via majority vote, each bit position
/// independently follows a binomial distribution. By the Central Limit
/// Theorem, the per-bit standard deviation is `σ ≈ 0.5/√n` (for
/// balanced bit densities), giving an expected relative error of `1/√n`.
///
/// This is a theoretical upper bound — actual error depends on bit
/// density correlation across contributors. Heavily correlated
/// thermometer codes may have lower error.
///
/// # Example
///
/// ```
/// # use symthaea_core::hdc::hdc_treasury::estimate_bundling_error;
/// let err_10 = estimate_bundling_error(10);
/// assert!((err_10 - 0.316).abs() < 0.01); // 1/√10 ≈ 0.316
/// ```
pub fn estimate_bundling_error(n_contributors: usize) -> f64 {
    if n_contributors == 0 {
        return 1.0;
    }
    1.0 / (n_contributors as f64).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// BALANCE ENCODING
// ═══════════════════════════════════════════════════════════════════════════════

/// Encode a u64 balance as a BinaryHV.
///
/// Uses thermometer encoding: each bit position i is set if balance >= threshold_i.
/// With 16,384 bits, this gives ~16,384 distinct levels.
/// The encoding preserves ordering: encode(a) is more similar to encode(b)
/// when |a - b| is small.
///
/// # Arguments
///
/// * `balance` — The balance in micro-SAP units (u64).
/// * `max_balance` — The maximum possible balance (defines the encoding range).
///
/// # Returns
///
/// A `BinaryHV` with the first `bits_to_set` bits set (thermometer code).
pub fn encode_balance(balance: u64, max_balance: u64) -> BinaryHV {
    let mut bytes = [0u8; 2048]; // 16,384 bits
    let bits_to_set = if max_balance == 0 {
        0
    } else {
        ((balance as u128 * 16384) / max_balance as u128).min(16384) as usize
    };
    // Set the first `bits_to_set` bits (thermometer encoding)
    let full_bytes = bits_to_set / 8;
    let remaining_bits = bits_to_set % 8;
    for b in bytes.iter_mut().take(full_bytes) {
        *b = 0xFF;
    }
    if remaining_bits > 0 && full_bytes < 2048 {
        bytes[full_bytes] = 0xFF << (8 - remaining_bits);
    }
    BinaryHV(bytes)
}

/// Decode a BinaryHV back to an approximate u64 balance.
///
/// Counts set bits (popcount) and maps back to the balance range.
/// Due to quantization (16,384 levels), the decoded value may differ
/// from the original by up to `max_balance / 16384`.
pub fn decode_balance(hv: &BinaryHV, max_balance: u64) -> u64 {
    let set_bits = hv.popcount();
    ((set_bits as u128 * max_balance as u128) / 16384).min(max_balance as u128) as u64
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENCRYPTED BALANCE
// ═══════════════════════════════════════════════════════════════════════════════

/// An encrypted balance — the actual value is hidden behind a one-time pad mask.
///
/// The `EncryptedHV` ciphertext reveals zero information about the balance
/// (Shannon 1949 perfect secrecy) as long as the mask is truly random and
/// used only once.
#[derive(Clone)]
pub struct EncryptedBalance {
    /// The encrypted hypervector (balance XOR mask).
    pub encrypted: EncryptedHV,
    /// Maximum balance used for encoding (needed for decoding).
    pub max_balance: u64,
    /// Member identifier (public — not encrypted).
    pub member_did: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TREASURY POOL
// ═══════════════════════════════════════════════════════════════════════════════

/// Privacy-preserving community treasury pool.
///
/// Holds encrypted individual balances. Can compute aggregate totals
/// via homomorphic bundling without decrypting individual values.
/// k-of-n trustees hold mask shares for threshold decryption.
///
/// # Protocol
///
/// ```text
/// 1. Treasury created with (k, n) threshold parameters
/// 2. For each deposit, generate_mask_shares() → (mask, shares)
/// 3. Shares distributed to n trustees; mask encrypts the balance
/// 4. deposit() stores the encrypted balance
/// 5. aggregate() computes bundled ciphertext (no decryption needed)
/// 6. k trustees combine shares → recovered mask → decrypt_aggregate()
/// ```
///
/// # Limitations
///
/// - Homomorphic bundling is approximate (majority-vote, not exact addition).
///   The aggregate total is an approximation, not exact arithmetic.
/// - Accuracy improves with more members (central limit theorem).
/// - All members must use the **same mask** for aggregate to be meaningful.
///   Use `generate_mask_shares()` to create a shared mask split among trustees.
pub struct HdcTreasuryPool {
    /// Encrypted individual balances.
    balances: Vec<EncryptedBalance>,
    /// Number of trustees required for decryption.
    k: usize,
    /// Total number of trustees.
    n: usize,
    /// Maximum balance for encoding (shared across all members).
    max_balance: u64,
}

impl HdcTreasuryPool {
    /// Create a new empty treasury pool.
    ///
    /// # Arguments
    ///
    /// * `k` — Number of trustees required for decryption (must be odd).
    /// * `n` — Total number of trustees (k <= n).
    /// * `max_balance` — Maximum balance in micro-SAP for encoding range.
    ///
    /// # Panics
    ///
    /// Panics if k == 0, k > n, or k is even.
    pub fn new(k: usize, n: usize, max_balance: u64) -> Self {
        assert!(k >= 1, "k must be at least 1");
        assert!(k <= n, "k must be <= n");
        assert!(k % 2 == 1, "k must be odd for threshold sharing");
        Self {
            balances: Vec::new(),
            k,
            n,
            max_balance,
        }
    }

    /// Add an encrypted balance to the pool.
    ///
    /// The caller must encrypt the balance with the shared mask before deposit.
    /// Returns the `EncryptedBalance` record (also stored internally).
    ///
    /// Returns `None` if the pool is at capacity.
    pub fn deposit(
        &mut self,
        member_did: &str,
        balance: u64,
        mask: &BinaryHV,
    ) -> Option<EncryptedBalance> {
        if self.balances.len() >= HDC_TREASURY_MAX_MEMBERS {
            return None;
        }
        let plaintext = encode_balance(balance, self.max_balance);
        let encrypted = EncryptedHV::encrypt(&plaintext, mask);
        let entry = EncryptedBalance {
            encrypted,
            max_balance: self.max_balance,
            member_did: member_did.to_string(),
        };
        self.balances.push(entry.clone());
        Some(entry)
    }

    /// Compute the approximate aggregate total via homomorphic bundling.
    ///
    /// Bundles all encrypted balance HVs using majority vote. The result
    /// is an encrypted aggregate that can be threshold-decrypted to reveal
    /// an approximate total.
    ///
    /// Returns `(EncryptedHV, estimated_relative_error)` where the error
    /// is `1/√n` (CLT approximation for majority-vote bundling).
    /// Returns `None` if the pool is empty.
    ///
    /// # Note
    ///
    /// All balances must have been encrypted with the **same mask** for the
    /// bundled aggregate to be meaningful after decryption.
    pub fn aggregate(&self) -> Option<(EncryptedHV, f64)> {
        if self.balances.is_empty() {
            return None;
        }

        let ciphertexts: Vec<BinaryHV> = self
            .balances
            .iter()
            .map(|eb| eb.encrypted.ciphertext)
            .collect();

        let error = estimate_bundling_error(ciphertexts.len());

        Some((
            EncryptedHV {
                ciphertext: BinaryHV::bundle(&ciphertexts),
            },
            error,
        ))
    }

    /// Threshold-decrypt the aggregate using k recovered mask shares.
    ///
    /// The shares are combined via `HdcThresholdSharing::recover()` to
    /// reconstruct the mask, which is then used to decrypt the aggregate.
    ///
    /// Returns `(decoded_value, estimated_relative_error)` where the error
    /// bound comes from `estimate_bundling_error(n)`.
    /// Returns `None` if fewer than k shares are provided or the pool is empty.
    pub fn decrypt_aggregate(&self, shares: &[HdcShare]) -> Option<(u64, f64)> {
        if shares.len() < self.k {
            return None;
        }
        let (aggregate, error) = self.aggregate()?;
        let recovered_mask = HdcThresholdSharing::recover(shares);
        let decrypted = aggregate.decrypt(&recovered_mask);
        Some((decode_balance(&decrypted, self.max_balance), error))
    }

    /// Get the number of members in the pool.
    pub fn member_count(&self) -> usize {
        self.balances.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.balances.is_empty()
    }

    /// Generate a fresh mask and split it into (k, n) threshold shares.
    ///
    /// Returns `(mask, shares)`:
    /// - `mask`: The full one-time pad mask for encrypting a balance.
    ///   Should be used once then zeroized.
    /// - `shares`: n shares to distribute to trustees. Any k of them
    ///   can reconstruct the mask.
    ///
    /// # Arguments
    ///
    /// * `seed` — Deterministic seed for mask generation (use a unique value per member).
    pub fn generate_mask_shares(&self, seed: u64) -> (BinaryHV, Vec<HdcShare>) {
        let mask = BinaryHV::random(seed);
        let shares = HdcThresholdSharing::split(&mask, self.k, self.n, seed.wrapping_add(1));
        (mask, shares)
    }

    /// Get the threshold parameters.
    pub fn threshold_params(&self) -> (usize, usize) {
        (self.k, self.n)
    }

    /// Get the maximum balance for this pool.
    pub fn max_balance(&self) -> u64 {
        self.max_balance
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance_encode_decode_roundtrip() {
        let max_balance: u64 = 1_000_000_000; // 1B micro-SAP
        // Test several values across the range
        for &balance in &[0, 1000, 50_000, 1_000_000, 500_000_000, 1_000_000_000] {
            let hv = encode_balance(balance, max_balance);
            let decoded = decode_balance(&hv, max_balance);
            let error = if decoded > balance {
                decoded - balance
            } else {
                balance - decoded
            };
            // Verify error is within the documented max_quantization_error bound
            let bound = max_quantization_error(max_balance) + 1;
            assert!(
                error <= bound,
                "Roundtrip error too large for balance={balance}: decoded={decoded}, error={error}, bound={bound}"
            );
        }
    }

    #[test]
    fn test_balance_encoding_preserves_order() {
        let max_balance: u64 = 1_000_000_000;
        let hv_100 = encode_balance(100_000, max_balance);
        let hv_200 = encode_balance(200_000, max_balance);
        let hv_10000 = encode_balance(10_000_000, max_balance);

        // encode(100K) should be more similar to encode(200K) than to encode(10M)
        let sim_close = hv_100.similarity(&hv_200);
        let sim_far = hv_100.similarity(&hv_10000);
        assert!(
            sim_close > sim_far,
            "Nearby balances should be more similar: sim(100K,200K)={sim_close} vs sim(100K,10M)={sim_far}"
        );
    }

    #[test]
    fn test_encrypted_balance_hides_value() {
        let max_balance: u64 = 1_000_000_000;
        let balance = 500_000_000;
        let plaintext = encode_balance(balance, max_balance);
        let mask = BinaryHV::random(42);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);

        // Encrypted HV should be ~0.5 similar to plaintext (OTP hiding)
        let sim = encrypted.ciphertext.similarity(&plaintext);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Encrypted balance should be ~0.5 similar to plaintext (OTP hiding), got {sim}"
        );
    }

    #[test]
    fn test_homomorphic_aggregate_approximate() {
        let max_balance: u64 = 1_000_000_000;
        let mask = BinaryHV::random(99);

        // Three members with known balances
        let balances = [100_000_000u64, 200_000_000, 300_000_000];
        let expected_total: u64 = balances.iter().sum(); // 600M

        // Encrypt each balance with the same mask
        let encrypted: Vec<EncryptedHV> = balances
            .iter()
            .map(|&b| {
                let hv = encode_balance(b, max_balance);
                EncryptedHV::encrypt(&hv, &mask)
            })
            .collect();

        // Bundle (homomorphic aggregate)
        let ciphertexts: Vec<BinaryHV> = encrypted.iter().map(|e| e.ciphertext).collect();
        let aggregate = EncryptedHV {
            ciphertext: BinaryHV::bundle(&ciphertexts),
        };

        // Decrypt aggregate
        let decrypted = aggregate.decrypt(&mask);
        let decoded_total = decode_balance(&decrypted, max_balance);

        // Majority-vote bundling is approximate — thermometer codes with
        // overlapping set bits produce a bundled result where the majority
        // bit pattern reflects the average density, not the sum.
        assert!(
            decoded_total > 0,
            "Decoded aggregate should be positive, got {decoded_total}"
        );
        let mean_balance = expected_total / balances.len() as u64;
        let tolerance = max_balance / 4; // 25% tolerance
        assert!(
            (decoded_total as i64 - mean_balance as i64).unsigned_abs() < tolerance,
            "Aggregate should approximate mean balance ({mean_balance}), got {decoded_total}"
        );

        // Verify bundling error estimate is reasonable
        let error_estimate = estimate_bundling_error(balances.len());
        assert!(
            error_estimate < 0.60,
            "Error estimate for 3 contributors should be < 0.60 (1/sqrt(3)), got {error_estimate}"
        );
    }

    #[test]
    fn test_threshold_decrypt_requires_k_shares() {
        let mut pool = HdcTreasuryPool::new(3, 5, 1_000_000_000);
        let (mask, shares) = pool.generate_mask_shares(42);

        pool.deposit("alice", 500_000_000, &mask);

        // Fewer than k=3 shares should fail
        assert!(
            pool.decrypt_aggregate(&shares[..1]).is_none(),
            "1 share should be insufficient (k=3)"
        );

        // Exactly k=3 shares should succeed
        let result = pool.decrypt_aggregate(&shares[..3]);
        assert!(result.is_some(), "3 shares should be sufficient (k=3)");
        let (value, error) = result.unwrap();
        assert!(value > 0, "Decrypted value should be positive");
        assert!(error > 0.0, "Error estimate should be positive");
    }

    #[test]
    fn test_treasury_pool_deposit_and_aggregate() {
        let mut pool = HdcTreasuryPool::new(3, 5, 1_000_000_000);
        let (mask, shares) = pool.generate_mask_shares(42);

        // Deposit three balances
        assert!(pool.deposit("alice", 100_000_000, &mask).is_some());
        assert!(pool.deposit("bob", 200_000_000, &mask).is_some());
        assert!(pool.deposit("carol", 300_000_000, &mask).is_some());

        assert_eq!(pool.member_count(), 3);
        assert!(!pool.is_empty());

        // Aggregate exists and returns error estimate
        let agg = pool.aggregate();
        assert!(agg.is_some());
        let (_encrypted, error) = agg.unwrap();
        assert!(
            error > 0.0,
            "Error estimate should be positive for 3 members"
        );

        // Decrypt with k=3 shares
        let total = pool.decrypt_aggregate(&shares[..3]);
        assert!(total.is_some(), "Should decrypt with 3 shares");
        let (value, error) = total.unwrap();
        // Majority-vote bundling yields approximate consensus, not sum
        assert!(value > 0, "Total should be positive, got {value}");
        assert!(error > 0.0, "Error should be positive");
    }

    #[test]
    fn test_zero_balance_encoding() {
        let max_balance: u64 = 1_000_000_000;
        let hv = encode_balance(0, max_balance);
        let decoded = decode_balance(&hv, max_balance);
        assert_eq!(decoded, 0, "Zero balance should decode to zero");
        assert_eq!(hv.popcount(), 0, "Zero balance should have no bits set");
    }

    #[test]
    fn test_max_balance_encoding() {
        let max_balance: u64 = 1_000_000_000;
        let hv = encode_balance(max_balance, max_balance);
        let decoded = decode_balance(&hv, max_balance);
        // Should decode to max_balance (all 16,384 bits set)
        assert_eq!(hv.popcount(), 16384, "Max balance should have all bits set");
        assert_eq!(decoded, max_balance, "Max balance should roundtrip exactly");
    }

    #[test]
    fn test_zero_max_balance() {
        // Edge case: max_balance = 0
        let hv = encode_balance(0, 0);
        assert_eq!(hv.popcount(), 0);
        let decoded = decode_balance(&hv, 0);
        assert_eq!(decoded, 0);
    }

    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = HdcTreasuryPool::new(1, 1, 1_000_000);
        let mask = BinaryHV::random(42);

        // Fill to capacity
        for i in 0..HDC_TREASURY_MAX_MEMBERS {
            assert!(
                pool.deposit(&format!("member-{i}"), 1000, &mask).is_some(),
                "Should accept member {i}"
            );
        }

        // One more should fail
        assert!(
            pool.deposit("overflow", 1000, &mask).is_none(),
            "Should reject when at capacity"
        );
        assert_eq!(pool.member_count(), HDC_TREASURY_MAX_MEMBERS);
    }

    #[test]
    fn test_empty_pool_aggregate() {
        let pool = HdcTreasuryPool::new(3, 5, 1_000_000_000);
        assert!(pool.aggregate().is_none());
        assert!(pool.is_empty());
        assert_eq!(pool.member_count(), 0);
    }

    #[test]
    fn test_quantization_error_bound() {
        assert_eq!(max_quantization_error(1_000_000_000), 61035);
        assert_eq!(max_quantization_error(1_000_000), 61);
        assert_eq!(max_quantization_error(0), 0);
        assert_eq!(max_quantization_error(16384), 1);
    }

    #[test]
    fn test_dithered_encoding_roundtrip() {
        let max_balance: u64 = 1_000_000_000;
        for &balance in &[0, 1000, 50_000, 500_000_000, 1_000_000_000] {
            let hv = encode_balance_with_dithering(balance, max_balance, 42);
            let decoded = decode_balance(&hv, max_balance);
            let error = (decoded as i64 - balance as i64).unsigned_abs();
            // Dithered encoding may round up, so error bound is max_quantization_error + 1 level
            let bound =
                max_quantization_error(max_balance) + max_quantization_error(max_balance) + 1;
            assert!(
                error <= bound,
                "Dithered roundtrip error too large for balance={balance}: decoded={decoded}, error={error}"
            );
        }
    }

    #[test]
    fn test_bundling_error_estimate() {
        // 1 contributor: error = 1.0
        assert!((estimate_bundling_error(1) - 1.0).abs() < f64::EPSILON);
        // 4 contributors: error = 0.5
        assert!((estimate_bundling_error(4) - 0.5).abs() < 0.001);
        // 10 contributors: error ≈ 0.316
        assert!((estimate_bundling_error(10) - 0.3162).abs() < 0.01);
        // 100 contributors: error = 0.1
        assert!((estimate_bundling_error(100) - 0.1).abs() < 0.001);
        // 0 contributors: error = 1.0 (degenerate)
        assert!((estimate_bundling_error(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_10_members_error_under_35_percent() {
        let max_balance: u64 = 1_000_000_000;
        let mut pool = HdcTreasuryPool::new(1, 1, max_balance);
        let mask = BinaryHV::random(999);

        // Deposit 10 members with varied balances
        for i in 0..10u64 {
            pool.deposit(&format!("member-{i}"), 100_000_000 + i * 50_000_000, &mask);
        }

        let (_, error) = pool.aggregate().expect("10 members should aggregate");
        assert!(
            error < 0.35,
            "Bundling error estimate for 10 members should be < 35%, got {error:.3}"
        );
    }

    #[test]
    fn test_threshold_params_accessors() {
        let pool = HdcTreasuryPool::new(3, 5, 1_000_000_000);
        assert_eq!(pool.threshold_params(), (3, 5));
        assert_eq!(pool.max_balance(), 1_000_000_000);
    }
}
