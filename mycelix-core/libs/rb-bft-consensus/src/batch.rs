// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Batch Signature Verification
//!
//! This module provides efficient batch verification for ed25519 signatures.
//! Batch verification can be significantly faster than individual verification
//! when verifying many signatures (typically 2-3x faster for 100+ signatures).
//!
//! ## How It Works
//!
//! Ed25519 batch verification uses random linear combination:
//! - Instead of checking each signature individually: R_i + H(R_i, A_i, m_i) * A_i = s_i * G
//! - We compute: sum(z_i * (R_i + H_i * A_i)) = sum(z_i * s_i) * G
//! - Where z_i are random scalars
//!
//! This is faster because we only do one expensive base-point multiplication.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rb_bft_consensus::batch::{BatchVerifier, SignatureEntry};
//!
//! let mut batch = BatchVerifier::new();
//!
//! // Add signatures to verify
//! for (pubkey, signature, message) in signatures {
//!     batch.add(&pubkey, &signature, message);
//! }
//!
//! // Verify all at once
//! match batch.verify() {
//!     Ok(()) => println!("All {} signatures valid", batch.len()),
//!     Err(e) => println!("Batch verification failed: {}", e),
//! }
//! ```

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::error::{ConsensusError, ConsensusResult};
use crate::crypto::ConsensusSignature;

/// A single entry in the batch verifier
#[derive(Clone)]
pub struct SignatureEntry {
    /// The public key to verify against
    pub public_key: [u8; 32],
    /// The signature bytes
    pub signature: [u8; 64],
    /// The message that was signed
    pub message: Vec<u8>,
    /// Optional label for error reporting
    pub label: Option<String>,
}

impl SignatureEntry {
    /// Create a new signature entry
    pub fn new(public_key: [u8; 32], signature: [u8; 64], message: Vec<u8>) -> Self {
        Self {
            public_key,
            signature,
            message,
            label: None,
        }
    }

    /// Create with a label
    pub fn with_label(
        public_key: [u8; 32],
        signature: [u8; 64],
        message: Vec<u8>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            public_key,
            signature,
            message,
            label: Some(label.into()),
        }
    }

    /// Create from a consensus signature
    pub fn from_consensus_signature(
        sig: &ConsensusSignature,
        message: Vec<u8>,
    ) -> ConsensusResult<Self> {
        let public_key: [u8; 32] = sig.signer_pubkey.as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Public key must be 32 bytes".to_string(),
            })?;

        let signature: [u8; 64] = sig.signature.as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Signature must be 64 bytes".to_string(),
            })?;

        Ok(Self {
            public_key,
            signature,
            message,
            label: None,
        })
    }
}

/// Result of batch verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerificationResult {
    /// Total signatures verified
    pub total_count: usize,
    /// Number of valid signatures
    pub valid_count: usize,
    /// Number of invalid signatures
    pub invalid_count: usize,
    /// Total verification time
    #[serde(skip)]
    pub total_time: Duration,
    /// Time in milliseconds for serialization
    pub time_ms: u64,
    /// Whether all signatures are valid
    pub all_valid: bool,
    /// Labels of failed signatures (if any)
    pub failed_labels: Vec<String>,
    /// Per-entry validity (true = valid, false = invalid)
    #[serde(default)]
    pub entry_validity: Vec<bool>,
    /// Per-entry error messages (empty string = valid)
    #[serde(default)]
    pub entry_errors: Vec<String>,
}

impl BatchVerificationResult {
    /// Calculate the speedup compared to individual verification
    pub fn speedup(&self, individual_time: Duration) -> f64 {
        if self.total_time.as_nanos() == 0 {
            return 1.0;
        }
        individual_time.as_nanos() as f64 / self.total_time.as_nanos() as f64
    }

    /// Average time per signature
    pub fn avg_time_per_sig(&self) -> Duration {
        if self.total_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.total_count as u32
        }
    }

    /// Check if entry at index is valid
    pub fn is_valid_by_index(&self, index: usize) -> bool {
        self.entry_validity.get(index).copied().unwrap_or(false)
    }

    /// Get error message for entry at index (None if valid)
    pub fn get_error_by_index(&self, index: usize) -> Option<String> {
        self.entry_errors.get(index).and_then(|e| {
            if e.is_empty() {
                None
            } else {
                Some(e.clone())
            }
        })
    }
}

/// Batch signature verifier
#[derive(Default)]
pub struct BatchVerifier {
    entries: Vec<SignatureEntry>,
}

impl BatchVerifier {
    /// Create a new empty batch verifier
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add a signature entry to the batch
    pub fn add(&mut self, entry: SignatureEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    /// Add a signature directly
    pub fn add_signature(
        &mut self,
        public_key: &[u8; 32],
        signature: &[u8; 64],
        message: &[u8],
    ) -> &mut Self {
        self.entries.push(SignatureEntry::new(
            *public_key,
            *signature,
            message.to_vec(),
        ));
        self
    }

    /// Add a signature with label
    pub fn add_signature_labeled(
        &mut self,
        public_key: &[u8; 32],
        signature: &[u8; 64],
        message: &[u8],
        label: impl Into<String>,
    ) -> &mut Self {
        self.entries.push(SignatureEntry::with_label(
            *public_key,
            *signature,
            message.to_vec(),
            label,
        ));
        self
    }

    /// Add a consensus signature
    pub fn add_consensus_signature(
        &mut self,
        sig: &ConsensusSignature,
        message: &[u8],
    ) -> ConsensusResult<&mut Self> {
        let entry = SignatureEntry::from_consensus_signature(sig, message.to_vec())?;
        self.entries.push(entry);
        Ok(self)
    }

    /// Get the number of signatures in the batch
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Verify all signatures in the batch using ed25519 batch verification
    ///
    /// This is typically faster than individual verification for batches > 4 signatures.
    pub fn verify(&self) -> ConsensusResult<BatchVerificationResult> {
        if self.entries.is_empty() {
            return Ok(BatchVerificationResult {
                total_count: 0,
                valid_count: 0,
                invalid_count: 0,
                total_time: Duration::ZERO,
                time_ms: 0,
                all_valid: true,
                failed_labels: Vec::new(),
                entry_validity: Vec::new(),
                entry_errors: Vec::new(),
            });
        }

        let start = Instant::now();

        // Prepare for batch verification
        let mut messages: Vec<&[u8]> = Vec::with_capacity(self.entries.len());
        let mut signatures: Vec<Signature> = Vec::with_capacity(self.entries.len());
        let mut verifying_keys: Vec<VerifyingKey> = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let verifying_key = VerifyingKey::from_bytes(&entry.public_key)
                .map_err(|e| ConsensusError::InvalidSignature {
                    reason: format!("Invalid public key: {}", e),
                })?;

            let signature = Signature::from_bytes(&entry.signature);

            messages.push(&entry.message);
            signatures.push(signature);
            verifying_keys.push(verifying_key);
        }

        // Perform batch verification
        let result = ed25519_dalek::verify_batch(&messages, &signatures, &verifying_keys);

        let total_time = start.elapsed();

        match result {
            Ok(()) => {
                let count = self.entries.len();
                Ok(BatchVerificationResult {
                    total_count: count,
                    valid_count: count,
                    invalid_count: 0,
                    total_time,
                    time_ms: total_time.as_millis() as u64,
                    all_valid: true,
                    failed_labels: Vec::new(),
                    entry_validity: vec![true; count],
                    entry_errors: vec![String::new(); count],
                })
            }
            Err(_) => {
                // Batch verification failed - fall back to individual verification
                // to identify which signatures failed
                self.verify_individual_fallback(start.elapsed())
            }
        }
    }

    /// Verify each signature individually (slower but provides detailed results)
    pub fn verify_individual(&self) -> ConsensusResult<BatchVerificationResult> {
        let start = Instant::now();
        self.verify_individual_fallback(Duration::ZERO)
            .map(|mut r| {
                r.total_time = start.elapsed();
                r.time_ms = r.total_time.as_millis() as u64;
                r
            })
    }

    /// Internal: verify individually to find failed signatures
    fn verify_individual_fallback(
        &self,
        initial_time: Duration,
    ) -> ConsensusResult<BatchVerificationResult> {
        let start = Instant::now();
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut failed_labels = Vec::new();
        let mut entry_validity = Vec::with_capacity(self.entries.len());
        let mut entry_errors = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let verifying_key = match VerifyingKey::from_bytes(&entry.public_key) {
                Ok(k) => k,
                Err(e) => {
                    invalid_count += 1;
                    if let Some(ref label) = entry.label {
                        failed_labels.push(label.clone());
                    }
                    entry_validity.push(false);
                    entry_errors.push(format!("Invalid public key: {}", e));
                    continue;
                }
            };

            let signature = Signature::from_bytes(&entry.signature);

            use ed25519_dalek::Verifier;
            match verifying_key.verify(&entry.message, &signature) {
                Ok(()) => {
                    valid_count += 1;
                    entry_validity.push(true);
                    entry_errors.push(String::new());
                }
                Err(e) => {
                    invalid_count += 1;
                    if let Some(ref label) = entry.label {
                        failed_labels.push(label.clone());
                    }
                    entry_validity.push(false);
                    entry_errors.push(format!("Signature verification failed: {}", e));
                }
            }
        }

        let total_time = initial_time + start.elapsed();

        Ok(BatchVerificationResult {
            total_count: self.entries.len(),
            valid_count,
            invalid_count,
            total_time,
            time_ms: total_time.as_millis() as u64,
            all_valid: invalid_count == 0,
            failed_labels,
            entry_validity,
            entry_errors,
        })
    }

    /// Verify and fail fast on first invalid signature
    pub fn verify_fail_fast(&self) -> ConsensusResult<()> {
        for entry in &self.entries {
            let verifying_key = VerifyingKey::from_bytes(&entry.public_key)
                .map_err(|e| ConsensusError::InvalidSignature {
                    reason: format!("Invalid public key: {}", e),
                })?;

            let signature = Signature::from_bytes(&entry.signature);

            use ed25519_dalek::Verifier;
            verifying_key.verify(&entry.message, &signature)
                .map_err(|e| ConsensusError::InvalidSignature {
                    reason: format!(
                        "Signature verification failed{}: {}",
                        entry.label.as_ref().map(|l| format!(" ({})", l)).unwrap_or_default(),
                        e
                    ),
                })?;
        }

        Ok(())
    }
}

/// Verify multiple votes in batch
pub fn batch_verify_votes(
    votes: &[(ConsensusSignature, Vec<u8>)],
) -> ConsensusResult<BatchVerificationResult> {
    let mut batch = BatchVerifier::with_capacity(votes.len());

    for (i, (sig, msg)) in votes.iter().enumerate() {
        let entry = SignatureEntry::from_consensus_signature(sig, msg.clone())?;
        batch.add(SignatureEntry::with_label(
            entry.public_key,
            entry.signature,
            entry.message,
            format!("vote_{}", i),
        ));
    }

    batch.verify()
}

/// Statistics for batch verification performance
#[derive(Debug, Clone, Default)]
pub struct BatchVerificationStats {
    /// Number of batch verifications performed
    pub batch_verifications: u64,
    /// Total signatures verified
    pub total_signatures: u64,
    /// Total time spent in batch verification
    pub total_time: Duration,
    /// Number of verification failures
    pub failures: u64,
    /// Average batch size
    pub avg_batch_size: f64,
}

impl BatchVerificationStats {
    /// Record a batch verification result
    pub fn record(&mut self, result: &BatchVerificationResult) {
        self.batch_verifications += 1;
        self.total_signatures += result.total_count as u64;
        self.total_time += result.total_time;
        if !result.all_valid {
            self.failures += 1;
        }
        self.avg_batch_size = self.total_signatures as f64 / self.batch_verifications as f64;
    }

    /// Get average verification time per signature
    pub fn avg_time_per_sig(&self) -> Duration {
        if self.total_signatures == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.total_signatures as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ValidatorKeypair;

    #[test]
    fn test_batch_verifier_empty() {
        let batch = BatchVerifier::new();
        let result = batch.verify().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 0);
    }

    #[test]
    fn test_batch_verifier_single() {
        let keypair = ValidatorKeypair::generate();
        let message = b"test message";
        let sig = keypair.sign(message);

        let mut batch = BatchVerifier::new();
        batch.add_consensus_signature(&sig, message).unwrap();

        let result = batch.verify().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 1);
        assert_eq!(result.valid_count, 1);
    }

    #[test]
    fn test_batch_verifier_multiple_valid() {
        let keypairs: Vec<ValidatorKeypair> = (0..10)
            .map(|_| ValidatorKeypair::generate())
            .collect();

        let mut batch = BatchVerifier::new();

        for (i, keypair) in keypairs.iter().enumerate() {
            let message = format!("message {}", i);
            let sig = keypair.sign(message.as_bytes());
            batch.add_signature_labeled(
                &keypair.public_key_bytes(),
                &sig.signature.as_slice().try_into().unwrap(),
                message.as_bytes(),
                format!("sig_{}", i),
            );
        }

        let result = batch.verify().unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 10);
        assert_eq!(result.valid_count, 10);
        assert!(result.failed_labels.is_empty());
    }

    #[test]
    fn test_batch_verifier_with_invalid() {
        let keypair = ValidatorKeypair::generate();
        let wrong_keypair = ValidatorKeypair::generate();

        let mut batch = BatchVerifier::new();

        // Add valid signature
        let msg1 = b"message 1";
        let sig1 = keypair.sign(msg1);
        batch.add_signature_labeled(
            &keypair.public_key_bytes(),
            &sig1.signature.as_slice().try_into().unwrap(),
            msg1,
            "valid_sig",
        );

        // Add invalid signature (wrong public key)
        let msg2 = b"message 2";
        let sig2 = keypair.sign(msg2);
        batch.add_signature_labeled(
            &wrong_keypair.public_key_bytes(), // Wrong key!
            &sig2.signature.as_slice().try_into().unwrap(),
            msg2,
            "invalid_sig",
        );

        let result = batch.verify().unwrap();

        assert!(!result.all_valid);
        assert_eq!(result.total_count, 2);
        assert_eq!(result.valid_count, 1);
        assert_eq!(result.invalid_count, 1);
        assert!(result.failed_labels.contains(&"invalid_sig".to_string()));
    }

    #[test]
    fn test_batch_verifier_fail_fast() {
        let keypair = ValidatorKeypair::generate();
        let wrong_keypair = ValidatorKeypair::generate();

        let mut batch = BatchVerifier::new();

        // Add valid signature first
        let msg1 = b"message 1";
        let sig1 = keypair.sign(msg1);
        batch.add_signature(
            &keypair.public_key_bytes(),
            &sig1.signature.as_slice().try_into().unwrap(),
            msg1,
        );

        // Add invalid signature
        let msg2 = b"message 2";
        let sig2 = keypair.sign(msg2);
        batch.add_signature_labeled(
            &wrong_keypair.public_key_bytes(),
            &sig2.signature.as_slice().try_into().unwrap(),
            msg2,
            "bad_sig",
        );

        let result = batch.verify_fail_fast();
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_verify_votes() {
        let keypairs: Vec<ValidatorKeypair> = (0..5)
            .map(|_| ValidatorKeypair::generate())
            .collect();

        let votes: Vec<(ConsensusSignature, Vec<u8>)> = keypairs.iter()
            .enumerate()
            .map(|(i, kp)| {
                let msg = format!("vote_{}", i).into_bytes();
                let sig = kp.sign(&msg);
                (sig, msg)
            })
            .collect();

        let result = batch_verify_votes(&votes).unwrap();

        assert!(result.all_valid);
        assert_eq!(result.total_count, 5);
    }

    #[test]
    fn test_batch_verification_stats() {
        let mut stats = BatchVerificationStats::default();

        let result1 = BatchVerificationResult {
            total_count: 10,
            valid_count: 10,
            invalid_count: 0,
            total_time: Duration::from_millis(50),
            time_ms: 50,
            all_valid: true,
            failed_labels: Vec::new(),
            entry_validity: vec![true; 10],
            entry_errors: vec![String::new(); 10],
        };

        let result2 = BatchVerificationResult {
            total_count: 20,
            valid_count: 18,
            invalid_count: 2,
            total_time: Duration::from_millis(100),
            time_ms: 100,
            all_valid: false,
            failed_labels: vec!["sig_5".to_string(), "sig_12".to_string()],
            entry_validity: vec![true; 18].into_iter().chain(vec![false; 2]).collect(),
            entry_errors: vec![String::new(); 18].into_iter().chain(vec!["error".to_string(); 2]).collect(),
        };

        stats.record(&result1);
        stats.record(&result2);

        assert_eq!(stats.batch_verifications, 2);
        assert_eq!(stats.total_signatures, 30);
        assert_eq!(stats.failures, 1);
        assert!((stats.avg_batch_size - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_large_batch_performance() {
        let keypairs: Vec<ValidatorKeypair> = (0..100)
            .map(|_| ValidatorKeypair::generate())
            .collect();

        let mut batch = BatchVerifier::with_capacity(100);

        for (i, keypair) in keypairs.iter().enumerate() {
            let message = format!("message_{}", i);
            let sig = keypair.sign(message.as_bytes());
            batch.add_signature(
                &keypair.public_key_bytes(),
                &sig.signature.as_slice().try_into().unwrap(),
                message.as_bytes(),
            );
        }

        // Batch verification
        let batch_result = batch.verify().unwrap();

        // Individual verification
        let individual_result = batch.verify_individual().unwrap();

        assert!(batch_result.all_valid);
        assert!(individual_result.all_valid);

        // Batch should typically be faster for large batches
        // (though this can vary in tests due to system load)
        println!(
            "Batch: {:?}, Individual: {:?}",
            batch_result.total_time, individual_result.total_time
        );
    }
}
