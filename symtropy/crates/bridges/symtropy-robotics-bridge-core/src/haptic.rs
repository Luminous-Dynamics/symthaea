// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Haptic Oracle — Robotic Proprioceptive Attestation.
//!
//! Enables robots to act as mobile DeSci nodes, signing verifiable
//! tactile claims about their physical environment using joint-level
//! enclaves (ARM TrustZone / ATECC608A).

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A signed tactile observation from a specific robotic joint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticAttestation {
    pub joint_id: String,
    pub platform_id: String,
    /// Joint resistance / prediction error (Channel 5)
    pub physical_resistance: f64,
    pub hardware_signature: Vec<u8>,
    pub sensor_pubkey: [u8; 32],
}

#[async_trait]
pub trait HapticOracle {
    /// Capture and sign a physical encounter using a joint enclave.
    async fn sign_haptic_encounter(
        &self,
        joint_id: &str,
        resistance: f64,
    ) -> Result<HapticAttestation>;

    /// Check if the haptic encounter indicates a significant environmental anomaly.
    fn detect_tactile_surprise(&self, attestation: &HapticAttestation, expected: f64) -> bool {
        (attestation.physical_resistance - expected).abs() > 0.5
    }
}
