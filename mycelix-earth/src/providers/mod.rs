// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::aoi::Aoi;
use crate::evidence::{EarthDataSource, EvidencePacket};
use async_trait::async_trait;
use mycelix_desci_core::{EmpiricalAxis, LEMCube, MaterialityAxis, NormativeAxis};
use serde::{Deserialize, Serialize};
use tracing::info;

#[async_trait]
pub trait EarthProvider {
    async fn search_metadata(
        &self,
        aoi: &Aoi,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<Vec<String>>;
    /// Fuses multiple orbital and ground products into a single high-integrity EvidencePacket.
    async fn fetch_fused_evidence(&self, product_ids: &[String]) -> anyhow::Result<EvidencePacket>;
    async fn extract_feature(
        &self,
        product_ids: &[String],
        feature_name: &str,
    ) -> anyhow::Result<EvidencePacket>;
}

pub mod hardware;

pub use hardware::{HardwareOracle, PhysicalEnclaveProvider};

pub struct SentinelHubProvider {
    pub api_key: String,
}

#[async_trait]
impl EarthProvider for SentinelHubProvider {
    async fn search_metadata(
        &self,
        _aoi: &Aoi,
        _start: chrono::DateTime<chrono::Utc>,
        _end: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<Vec<String>> {
        // Mocking discovery of both S1 and S2 products
        Ok(vec![
            "S2L2A-2026-05-01-T34HDC".to_string(),  // Optical (Skin)
            "S1GRD-2026-05-01-IW_VVVH".to_string(), // Radar (Structure)
        ])
    }

    async fn fetch_fused_evidence(&self, product_ids: &[String]) -> anyhow::Result<EvidencePacket> {
        info!(
            "🛰️  Performing Multi-Modal Fusion: Sentinel-2 (NDVI) + Sentinel-1 (SAR Structure)..."
        );

        // In production:
        // 1. Fetch S2 L2A tile -> extract NDVI (Chlorophyll)
        // 2. Fetch S1 GRD tile -> extract Backscatter (Structural volume)
        // 3. Spatio-temporal alignment

        Ok(EvidencePacket {
            id: uuid::Uuid::new_v4(),
            source: EarthDataSource::OrbitalFusion,
            product_ids: product_ids.to_vec(),
            aoi_hash: "fused-aoi-hash".to_string(),
            observed_at: chrono::Utc::now(),
            processing_version: "fusion-v1.0-sar-optical".to_string(),
            feature_name: "HolisticRestorationIndex".to_string(),
            value: 0.88, // Combined score
            unit: "HRI".to_string(),
            uncertainty: 0.03, // Lower uncertainty due to multi-modal cross-check
            checksum: "fusion-checksum-01".to_string(),
            lem: LEMCube::new(
                EmpiricalAxis::E2PrivatelyVerifiable, // Still E2 until hardware-enclaved
                NormativeAxis::N1Communal,
                MaterialityAxis::M2Persistent,
            ),
            hardware_signature: None,
            sensor_pubkey: None,
            joules_consumed: 4.2, // Higher cost for fused processing
            somatic_witnesses: Vec::new(),
        })
    }

    async fn extract_feature(
        &self,
        product_ids: &[String],
        feature_name: &str,
    ) -> anyhow::Result<EvidencePacket> {
        // Mock implementation for v0
        Ok(EvidencePacket {
            id: uuid::Uuid::new_v4(),
            source: EarthDataSource::Sentinel2,
            product_ids: product_ids.to_vec(),
            aoi_hash: "mock-aoi-hash".to_string(),
            observed_at: chrono::Utc::now(),
            processing_version: "v0.1.0".to_string(),
            feature_name: feature_name.to_string(),
            value: 0.85, // High NDVI
            unit: "NDVI".to_string(),
            uncertainty: 0.05,
            checksum: "mock-checksum".to_string(),
            lem: LEMCube::new(
                EmpiricalAxis::E2PrivatelyVerifiable, // Legacy Feed (ESA)
                NormativeAxis::N1Communal,
                MaterialityAxis::M2Persistent,
            ),
            hardware_signature: None,
            sensor_pubkey: None,
            joules_consumed: 0.5, // Network transfer cost
            somatic_witnesses: Vec::new(),
        })
    }
}
