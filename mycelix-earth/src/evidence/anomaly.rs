// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! FEP Anomaly & Contradiction Engine
//!
//! Detects entropic contradictions between EvidencePackets using the
//! Free Energy Principle. Triggers [Epistemic Alerts] when new data
//! violates the established E4 consensus.

use crate::evidence::EvidencePacket;
use mycelix_desci_core::EmpiricalAxis;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicAlert {
    pub claim_id: uuid::Uuid,
    pub contradiction_id: uuid::Uuid,
    pub entropy_score: f64, // Surprise / Prediction Error
    pub thermodynamic_conflict_joules: f64,
    pub status: AlertStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    FreezingPayouts,
    AwaitingGroundTruth,
    LazinessAnomaly,
    Resolved,
}

pub struct FepDetector {
    /// Threshold for "surprise" before an alert is triggered
    pub surprise_threshold: f64,
    /// Maximum allowed similarity for peer reviews (anti-lazy)
    pub max_semantic_similarity: f64,
}

impl FepDetector {
    pub fn new(threshold: f64, max_sim: f64) -> Self {
        Self {
            surprise_threshold: threshold,
            max_semantic_similarity: max_sim,
        }
    }

    /// Detect if a review is just a "Semantic Echo" of the claim.
    pub fn detect_laziness(
        &self,
        similarity_score: f64,
        review_id: uuid::Uuid,
    ) -> Option<EpistemicAlert> {
        if similarity_score > self.max_semantic_similarity {
            warn!(
                "⚠️ [LAZINESS ANOMALY] Review {} is a semantic echo (Sim: {:.2})",
                review_id, similarity_score
            );

            Some(EpistemicAlert {
                claim_id: uuid::Uuid::nil(), // Alert is on the review, not the claim
                contradiction_id: review_id,
                entropy_score: similarity_score,
                thermodynamic_conflict_joules: 0.0,
                status: AlertStatus::LazinessAnomaly,
            })
        } else {
            None
        }
    }

    /// Analyze a new packet against an existing high-consensus claim.
    pub fn detect_contradiction(
        &self,
        new_packet: &EvidencePacket,
        existing_e4: &EvidencePacket,
    ) -> Option<EpistemicAlert> {
        // Only trigger against established E4 truth
        if existing_e4.lem.empirical != EmpiricalAxis::E4PubliclyReproducible {
            return None;
        }

        // Calculate Prediction Error (Entropy)
        // Simplified: absolute difference relative to uncertainty
        let diff = (new_packet.value - existing_e4.value).abs();
        let pooled_uncertainty =
            (new_packet.uncertainty.powi(2) + existing_e4.uncertainty.powi(2)).sqrt();
        let surprise = diff / pooled_uncertainty.max(0.001);

        if surprise > self.surprise_threshold {
            warn!(
                "⚠️ [FEP ANOMALY] High surprise detected: {:.2} (Threshold: {:.2})",
                surprise, self.surprise_threshold
            );

            Some(EpistemicAlert {
                claim_id: existing_e4.id,
                contradiction_id: new_packet.id,
                entropy_score: surprise,
                thermodynamic_conflict_joules: new_packet.joules_consumed
                    + existing_e4.joules_consumed,
                status: AlertStatus::FreezingPayouts,
            })
        } else {
            None
        }
    }
}

/// Autonomously handle the Epistemic Alert
pub fn handle_epistemic_alert(alert: &EpistemicAlert) {
    info!(
        "⚖️ [Phase 7] Autonomously FREEZING LARP Bounty for claim: {}",
        alert.claim_id
    );
    info!(
        "🔥 Thermodynamic conflict logged: {:.2} Joules of contradictory proof.",
        alert.thermodynamic_conflict_joules
    );

    // Issue Observational Pull
    info!(
        "📡 [Observational Pull] Requesting urgent ground-truth from nearby nodes for AOI conflict."
    );
}
