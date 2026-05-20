// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Secure Dreaming — Zero-Trust Architectural Evolution with Post-Quantum Attestation

use crate::evolutionary_scaffolder::EvolutionResult;
use crate::self_optimization::SelfOptimizationEngine;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey, SecretKey, SignedMessage};
use sha2::{Digest, Sha256};
use std::path::Path;

pub struct SecureDreamingEngine {
    inner_optimizer: SelfOptimizationEngine,
    // Enclave identity (Dilithium5 keypair)
    public_key: dilithium5::PublicKey,
    secret_key: dilithium5::SecretKey,
}

impl SecureDreamingEngine {
    pub fn new(project_root: std::path::PathBuf) -> Self {
        let (pk, sk) = dilithium5::keypair();
        Self {
            inner_optimizer: SelfOptimizationEngine::new(project_root),
            public_key: pk,
            secret_key: sk,
        }
    }

    pub fn dream_securely(
        &self,
        file_path: &Path,
        original_intent: &str,
    ) -> anyhow::Result<SecureDreamResult> {
        println!("🔒 Entering Secure Enclave (simulated TEE with Dilithium5)...");

        // Run full optimization + reflection inside the "enclave"
        let evolution_result = self.inner_optimizer.evolve_file(file_path)?;
        let success = evolution_result.success_score > 0.8;

        // === Post-Quantum Attestation ===
        let attestation_data = format!(
            "enclave_id: symthaea-tee-pqe-v1\n\
             timestamp: {}\n\
             file: {}\n\
             intent: {}\n\
             success: {}\n\
             score: {:.4}\n\
             reason: {}",
            chrono::Utc::now().to_rfc3339(),
            file_path.display(),
            original_intent,
            success,
            evolution_result.success_score,
            evolution_result.mutation_description
        );

        // 1. Generate SHA-256 manifest hash
        let mut hasher = Sha256::new();
        hasher.update(attestation_data.as_bytes());
        let report_hash = format!("{:x}", hasher.finalize());

        // 2. Sign the attestation with Dilithium5 (PQE)
        let signature = dilithium5::sign(attestation_data.as_bytes(), &self.secret_key);
        let signature_hex = hex::encode(signature.as_bytes());

        let attestation = format!(
            "TEE_PQ_ATTESTATION_REPORT\n\
             Hash: {}\n\
             Signature: {}\n\
             Algorithm: Dilithium5 (NIST Level 5)\n\
             Status: {}",
            report_hash,
            signature_hex,
            if success { "ACCEPTED" } else { "REJECTED" }
        );

        Ok(SecureDreamResult {
            success,
            new_score: evolution_result.success_score,
            attestation,
            pq_signature: signature_hex,
            enclave_pk: hex::encode(self.public_key.as_bytes()),
        })
    }
}

#[derive(Debug)]
pub struct SecureDreamResult {
    pub success: bool,
    pub new_score: f32,
    pub attestation: String,
    pub pq_signature: String,
    pub enclave_pk: String,
}
