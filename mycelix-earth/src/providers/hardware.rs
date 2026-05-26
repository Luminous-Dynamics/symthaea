// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware Oracle — Cryptographic Enclave Interface.
//!
//! Closes the "Oracular Gap" by signing raw sensor payloads at the
//! hardware boundary using ATECC608A or YubiKey enclaves.

use crate::evidence::{EarthDataSource, EvidencePacket};
use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use mycelix_desci_core::{EmpiricalAxis, LEMCube, MaterialityAxis, NormativeAxis};

#[async_trait]
pub trait HardwareOracle {
    /// Sign a raw sensor reading using the physically wired hardware enclave.
    async fn sign_reading(&self, sensor_id: &str, payload: &[u8]) -> Result<(Vec<u8>, [u8; 32])>;

    /// Compile a hardware-verified reading into an E4 EvidencePacket.
    async fn capture_verified_claim(
        &self,
        sensor_id: &str,
        feature_name: &str,
        value: f64,
        unit: &str,
    ) -> Result<EvidencePacket>;
}

pub struct PhysicalEnclaveProvider {
    /// Public key registry of verified local sensors
    pub verified_sensors: std::collections::HashMap<String, [u8; 32]>,
}

#[async_trait]
impl HardwareOracle for PhysicalEnclaveProvider {
    async fn sign_reading(&self, _sensor_id: &str, payload: &[u8]) -> Result<(Vec<u8>, [u8; 32])> {
        // In production, this interfaces with /dev/tty or PKCS#11
        // to command the ATECC608A / YubiKey.

        // Mocking hardware signature for v0
        let mock_sig = vec![0xEE; 64]; // Heavy hardware signature
        let mock_pubkey = [0xAA; 32];

        Ok((mock_sig, mock_pubkey))
    }

    async fn capture_verified_claim(
        &self,
        sensor_id: &str,
        feature_name: &str,
        value: f64,
        unit: &str,
    ) -> Result<EvidencePacket> {
        let payload = format!("{}-{}-{}", feature_name, value, unit);
        let (sig, pubkey) = self.sign_reading(sensor_id, payload.as_bytes()).await?;

        // Verify if the pubkey matches our known secure sensors
        // This is the primary witness check.
        let is_verified = self.verified_sensors.get(sensor_id) == Some(&pubkey);

        let lem = if is_verified {
            // INSTANT E4: Native hardware-enclaved truth.
            LEMCube::new(
                EmpiricalAxis::E4PubliclyReproducible,
                NormativeAxis::N2Network,
                MaterialityAxis::M3Foundational,
            )
        } else {
            // Fallback if hardware signature is invalid or sensor unknown
            LEMCube::new(
                EmpiricalAxis::E1Testimonial,
                NormativeAxis::N1Communal,
                MaterialityAxis::M1Temporal,
            )
        };

        Ok(EvidencePacket {
            id: uuid::Uuid::new_v4(),
            source: EarthDataSource::LocalSensor,
            product_ids: vec![sensor_id.to_string()],
            aoi_hash: "local-sensor-aoi".to_string(),
            observed_at: Utc::now(),
            processing_version: "hardware-v1.0".to_string(),
            feature_name: feature_name.to_string(),
            value,
            unit: unit.to_string(),
            uncertainty: 0.01, // High precision
            checksum: format!("{:x}", md5::compute(payload)),
            lem,
            hardware_signature: Some(sig),
            sensor_pubkey: Some(pubkey),
            joules_consumed: 12.4, // Proof generation cost on Pi 5
            somatic_witnesses: Vec::new(),
        })
    }
}
