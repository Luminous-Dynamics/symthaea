// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sovereign Bridge — Athena <-> Governance/Ledger Connector
//!
//! Provides a stable, decoupled API for Athena-Triage agents to interact
//! with the Governance and Ledger modules without needing direct dependency
//! on the volatile cognitive loop modules.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use mycelix_crypto::AlgorithmId;
use mycelix_crypto::envelope::{TaggedPublicKey, TaggedSignature};
use mycelix_crypto::traits::Verifier;

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Governance API error: {0}")]
    Governance(String),
    #[error("Cryptographic error: {0}")]
    Crypto(String),
    #[error("Verification failed")]
    VerificationFailed,
}

pub struct BridgeClient {
    governance_url: String,
}

impl BridgeClient {
    pub fn new(governance_url: String) -> Self {
        Self { governance_url }
    }

    /// Verify a threshold signature against the committee public key.
    pub fn verify_mitigation_authority(
        &self,
        content_hash: &[u8],
        signature_bytes: &[u8],
        committee_pk_bytes: &[u8],
    ) -> Result<bool, BridgeError> {
        let verifier = mycelix_crypto::pqc::dilithium::MlDsa65Verifier;
        let pk = TaggedPublicKey::new(AlgorithmId::MlDsa65, committee_pk_bytes.to_vec())
            .map_err(|e| BridgeError::Crypto(format!("{:?}", e)))?;
        let sig = TaggedSignature::new(AlgorithmId::MlDsa65, signature_bytes.to_vec())
            .map_err(|e| BridgeError::Crypto(format!("{:?}", e)))?;

        verifier
            .verify(&pk, content_hash, &sig)
            .map_err(|e| BridgeError::Crypto(format!("{:?}", e)))
    }

    pub async fn propose_emergency_mitigation(
        &self,
        title: &str,
        description: &str,
        target: &str,
    ) -> Result<(), BridgeError> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/propose", self.governance_url))
            .json(&serde_json::json!({
                "title": title,
                "description": description,
                "proposal_type": "Emergency",
                "author": "did:mycelix:athena",
                "actions": serde_json::json!({"action": "revoke_access", "target": target})
            }))
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(BridgeError::Governance(format!(
                "Proposal failed: {}",
                resp.status()
            )));
        }
        Ok(())
    }
}
