// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod anomaly;
pub mod decay;

use chrono::{DateTime, Utc};
use mycelix_desci_core::LEMCube;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarthDataSource {
    Sentinel1,
    Sentinel2,
    Landsat8,
    LocalSensor,
    CommunityReport,
    /// Fused Sentinel-1 (Radar) and Sentinel-2 (Optical) data
    OrbitalFusion,
    /// Mobile robotic tactile/proprioceptive attestation
    Tactile,
}

/// A human attestation to a physical event or reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaticWitness {
    pub agent: hdi::prelude::AgentPubKey,
    /// Biometric-locked signature from a mobile enclave
    pub biometric_signature: Vec<u8>,
    pub observed_at: DateTime<Utc>,
    /// Optional somatic feeling/confidence score (0.0 - 1.0)
    pub somatic_confidence: f64,
}

/// An Evidence Packet from Earth observation.
///
/// This is the foundational unit of the Earth Evidence Mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePacket {
    pub id: uuid::Uuid,
    pub source: EarthDataSource,
    /// List of source product IDs (e.g. Sentinel-2 L2A tile IDs)
    pub product_ids: Vec<String>,
    pub aoi_hash: String,
    pub observed_at: DateTime<Utc>,
    pub processing_version: String,
    pub feature_name: String,
    pub value: f64,
    pub unit: String,
    pub uncertainty: f64,
    pub checksum: String,
    /// LEM classification (Epistemic/Normative/Materiality)
    pub lem: LEMCube,
    /// Optional cryptographic signature from physical hardware enclave
    pub hardware_signature: Option<Vec<u8>>,
    /// Public key of the sensor that generated the signature
    pub sensor_pubkey: Option<[u8; 32]>,
    /// Energy consumed during capture and proving (Joules).
    pub joules_consumed: f64,
    /// Human attestations (Somatic Pulse)
    pub somatic_witnesses: Vec<SomaticWitness>,
}
