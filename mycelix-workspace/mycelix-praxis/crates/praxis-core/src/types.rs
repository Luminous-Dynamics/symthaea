// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared core types for Mycelix Praxis (Type 1 Civilization Substrate)
//!
//! Includes primitives for liquid learning, byzantine-tolerant FL, and
//! decentralized educational coordination.

use holo_hash::ActionHash;
use holochain_serialized_bytes::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// REPUTATION & CURRENCY (TEND)
// =============================================================================

/// Utility Voucher (TEND): Closed-loop reputational credit.
/// Explicitly NOT a cryptocurrency or speculative asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityVoucher {
    pub balance: u64,
    pub guild_id: String,               // Bound to a specific local cooperative
    pub is_transferable_external: bool, // Always false for regulatory compliance
}

/// Blind Audit: Cross-mesh verification of mastery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindAudit {
    pub spore_hash: ActionHash,
    pub auditor_node_id: String, // Anonymized external node
    pub result: AuditResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditResult {
    Verified,
    MaliciousCollusion, // Triggers group reputation slashing
    LowQuality,
}

/// Maintenance Escrow: Thermodynamic attrition fund.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceEscrow {
    pub hardware_id: String,
    pub accumulated_tend: u64,
    pub spare_part_target: u64,
}

/// Hive-Mind Procurement: Fragmented micro-purchases to avoid extortion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentedOrder {
    pub total_proposal_id: ActionHash,
    pub micro_purchases: Vec<MicroPurchase>,
    pub assembly_node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroPurchase {
    pub component_id: String,
    pub delivery_destination_hash: String, // Randomized student residence
    pub is_received: bool,
}

// =============================================================================
// COORDINATION & LAND
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydroPatch {
    pub burst_pipe_coordinate: String,
    pub clamp_type_id: String, // 3D-printed spec
    pub estimated_liters_saved: u64,
    pub patch_guild_did: String,
}

/// Resource Profile: Thermodynamic and material requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceProfile {
    pub class: String, // e.g. "DC_Storage_Cell"
    pub min_capacity_mah: Option<u32>,
    pub thermodynamic_threshold: Option<f32>,
    pub embodied_energy_joules: u64, // The MJ cost of the material
    pub common_scraps: Vec<String>,
}

/// Liquid Holocell: Dynamic HDC dimensionality for thermodynamic management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidHolocell {
    pub current_dimensionality: u32, // 2^14 to 2^16
    pub power_draw_watts: f32,
    pub semantic_resolution_score: f32,
}

/// Kinetic Signature: IMU-based proof of manual labor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticSignature {
    pub agent_did: String,
    pub task_type: String,   // e.g. "CEB_Press", "Hand_Sanding"
    pub motion_hash: String, // Cryptographic hash of sensor data
    pub repetition_count: u32,
    pub intensity_delta: f32,
}

/// Spore Ejection: Rapid evacuation of node state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeEjection {
    pub node_id: String,
    pub encrypted_state_blob: Vec<u8>,
    pub ejection_trigger: String, // e.g. "Raid_Detected", "Seismic_Collapse"
    pub target_uplink_id: String,
}

// =============================================================================
// FEDERATED LEARNING
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, SerializedBytes, PartialEq, Eq, Hash)]
pub struct RoundId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoundState {
    Active,
    Aggregating,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams {
    pub clip_norm: f32,
    pub epsilon: Option<f32>,
}

impl Default for PrivacyParams {
    fn default() -> Self {
        Self {
            clip_norm: 1.0,
            epsilon: None,
        }
    }
}

// =============================================================================
// IDENTITIES & HASHES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, SerializedBytes, PartialEq, Eq, Hash)]
pub struct ModelHash(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize, SerializedBytes, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, SerializedBytes, PartialEq, Eq, Hash)]
pub struct CourseId(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_id_creation() {
        let id = RoundId("round-123".to_string());
        assert_eq!(id.0, "round-123");
    }

    #[test]
    fn test_default_privacy_params() {
        let params = PrivacyParams::default();
        assert_eq!(params.clip_norm, 1.0);
        assert!(params.epsilon.is_none());
    }
}
