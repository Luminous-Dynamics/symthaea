// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic Memory Attestation
//!
//! BLAKE3 hashing of critical data structures at startup with periodic re-verification.
//! Catches static tampering (binary patching, memory corruption) but not dynamic
//! manipulation of values between hash checks.
//!
//! # Usage
//!
//! Structures implement the `Attestable` trait, then register with the `AttestationRegistry`.
//! The registry computes baseline hashes at registration and verifies them periodically.

use std::time::Instant;

/// Trait for structures that can produce a deterministic hash of their critical state.
///
/// Implementors must guarantee that `integrity_hash()` returns the same bytes for
/// the same logical state across calls (no floating-point non-determinism, no
/// pointer-dependent ordering).
pub trait Attestable {
    /// Compute BLAKE3 hash of the structure's integrity-critical bytes.
    fn integrity_hash(&self) -> [u8; 32];

    /// Human-readable name for the attestation record.
    fn attestation_name(&self) -> &'static str;
}

/// A single attestation record: what was attested, its baseline hash, and verification history.
#[derive(Debug, Clone)]
pub struct AttestationRecord {
    /// Human-readable name of the attested structure.
    pub name: &'static str,
    /// BLAKE3 hash computed at registration (baseline).
    pub baseline_hash: [u8; 32],
    /// Timestamp of baseline computation.
    pub baseline_time: Instant,
    /// Most recent verification result.
    pub last_verification: Option<VerificationResult>,
    /// Number of consecutive verification failures (for drift vs corruption detection).
    pub consecutive_failures: usize,
}

/// Result of a single verification check.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the current hash matched the baseline.
    pub passed: bool,
    /// The hash computed during verification.
    pub current_hash: [u8; 32],
    /// When the verification occurred.
    pub verified_at: Instant,
    /// Cycle number when verified.
    pub cycle: usize,
}

/// Registry of all attested structures.
///
/// Holds baseline hashes and provides bulk verification.
pub struct AttestationRegistry {
    /// Static records (name + baseline hash). The actual structures are not stored —
    /// re-hashing happens via registered closures.
    records: Vec<AttestationRecord>,
    /// Closures that re-compute the hash for each registered structure.
    /// Indexed in parallel with `records`.
    hashers: Vec<Box<dyn Fn() -> [u8; 32] + Send + Sync>>,
}

impl AttestationRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            hashers: Vec::new(),
        }
    }

    /// Register a structure for attestation.
    ///
    /// Computes the baseline hash immediately and stores a closure for re-verification.
    /// The closure must capture a reference or arc to the live structure so it can
    /// re-hash the current state on demand.
    /// Register a structure for attestation.
    ///
    /// Computes the baseline hash immediately and stores a closure for re-verification.
    /// The closure must capture a reference or arc to the live structure so it can
    /// re-hash the current state on demand.
    ///
    /// **Self-test**: immediately calls the hasher closure and asserts it matches
    /// `initial_hash`. This catches registration bugs (wrong closure, wrong slice)
    /// at startup rather than 101 cycles later.
    ///
    /// # Panics
    ///
    /// Panics if the hasher closure returns a different hash than `initial_hash`,
    /// indicating a registration bug.
    pub fn register(
        &mut self,
        name: &'static str,
        initial_hash: [u8; 32],
        hasher: Box<dyn Fn() -> [u8; 32] + Send + Sync>,
    ) {
        // Self-test: verify the hasher reproduces the baseline immediately
        let selftest_hash = hasher();
        assert_eq!(
            initial_hash, selftest_hash,
            "Attestation self-test FAILED for '{}': initial hash != hasher() at registration. \
             This is a registration bug — the closure doesn't match the provided hash.",
            name
        );
        self.records.push(AttestationRecord {
            name,
            baseline_hash: initial_hash,
            baseline_time: Instant::now(),
            last_verification: None,
            consecutive_failures: 0,
        });
        self.hashers.push(hasher);
    }

    /// Register without the self-test assertion.
    ///
    /// **Only for testing**: allows intentionally mismatched baseline/hasher to simulate
    /// tampering. Production code should always use `register()`.
    #[cfg(any(test, feature = "integrity"))]
    pub fn register_tampered(
        &mut self,
        name: &'static str,
        initial_hash: [u8; 32],
        hasher: Box<dyn Fn() -> [u8; 32] + Send + Sync>,
    ) {
        self.records.push(AttestationRecord {
            name,
            baseline_hash: initial_hash,
            baseline_time: Instant::now(),
            last_verification: None,
            consecutive_failures: 0,
        });
        self.hashers.push(hasher);
    }

    /// Register from an `Attestable` implementor.
    ///
    /// Note: This takes a snapshot hash at registration time. For runtime re-verification,
    /// use `register()` with a closure that can access the live structure.
    pub fn register_snapshot(&mut self, attestable: &dyn Attestable) {
        let hash = attestable.integrity_hash();
        let name = attestable.attestation_name();
        self.records.push(AttestationRecord {
            name,
            baseline_hash: hash,
            baseline_time: Instant::now(),
            last_verification: None,
            consecutive_failures: 0,
        });
        // Snapshot-only: re-verification returns the baseline (can't re-hash without access)
        let frozen = hash;
        self.hashers.push(Box::new(move || frozen));
    }

    /// Verify all registered structures against their baselines.
    ///
    /// Returns a list of failure descriptions (empty = all passed).
    pub fn verify_all(&mut self, cycle: usize) -> Vec<String> {
        let mut failures = Vec::new();
        for (i, hasher) in self.hashers.iter().enumerate() {
            let current_hash = hasher();
            let record = &mut self.records[i];
            let passed = current_hash == record.baseline_hash;
            record.last_verification = Some(VerificationResult {
                passed,
                current_hash,
                verified_at: Instant::now(),
                cycle,
            });
            if !passed {
                // Track consecutive failures for drift detection
                record.consecutive_failures += 1;
                let severity = if record.consecutive_failures >= 3 {
                    "CRITICAL"
                } else {
                    "DRIFT"
                };
                failures.push(format!(
                    "Attestation {} for '{}' (streak {}): baseline {:x?} != current {:x?}",
                    severity,
                    record.name,
                    record.consecutive_failures,
                    &record.baseline_hash[..8],
                    &current_hash[..8],
                ));
            } else {
                record.consecutive_failures = 0;
            }
        }
        failures
    }

    /// Number of registered attestations.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get all records for reporting.
    pub fn records(&self) -> &[AttestationRecord] {
        &self.records
    }
}

impl Default for AttestationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a BLAKE3 hash of a byte slice.
///
/// Convenience function for `Attestable` implementations.
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Compute a BLAKE3 hash of multiple f32 values (e.g., HDC vector components).
///
/// Converts each f32 to its little-endian byte representation for deterministic hashing.
pub fn blake3_hash_f32_slice(values: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for v in values {
        hasher.update(&v.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Compute a BLAKE3 hash of multiple f64 values.
pub fn blake3_hash_f64_slice(values: &[f64]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for v in values {
        hasher.update(&v.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyAttestable {
        data: Vec<f32>,
    }

    impl Attestable for DummyAttestable {
        fn integrity_hash(&self) -> [u8; 32] {
            blake3_hash_f32_slice(&self.data)
        }
        fn attestation_name(&self) -> &'static str {
            "test_dummy"
        }
    }

    #[test]
    fn test_attestable_deterministic() {
        let a = DummyAttestable {
            data: vec![1.0, 2.0, 3.0],
        };
        let h1 = a.integrity_hash();
        let h2 = a.integrity_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_attestable_different_data_different_hash() {
        let a = DummyAttestable {
            data: vec![1.0, 2.0, 3.0],
        };
        let b = DummyAttestable {
            data: vec![1.0, 2.0, 3.001],
        };
        assert_ne!(a.integrity_hash(), b.integrity_hash());
    }

    #[test]
    fn test_registry_register_and_verify() {
        let mut reg = AttestationRegistry::new();
        let hash = blake3_hash(b"hello");
        reg.register("test", hash, Box::new(move || blake3_hash(b"hello")));
        assert_eq!(reg.len(), 1);
        let failures = reg.verify_all(1);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_registry_detects_tampering() {
        let mut reg = AttestationRegistry::new();
        let baseline = blake3_hash(b"original");
        reg.register_tampered(
            "tampered",
            baseline,
            Box::new(move || blake3_hash(b"modified")),
        );
        let failures = reg.verify_all(1);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("DRIFT"));
    }

    #[test]
    fn test_snapshot_registration() {
        let a = DummyAttestable {
            data: vec![42.0, 99.0],
        };
        let mut reg = AttestationRegistry::new();
        reg.register_snapshot(&a);
        assert_eq!(reg.len(), 1);
        // Snapshot re-verification always passes (frozen hash)
        let failures = reg.verify_all(1);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_blake3_hash_f32_deterministic() {
        let data = [1.0f32, 2.0, 3.14159, -0.5];
        let h1 = blake3_hash_f32_slice(&data);
        let h2 = blake3_hash_f32_slice(&data);
        assert_eq!(h1, h2);
    }
}
