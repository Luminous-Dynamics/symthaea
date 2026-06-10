// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified epistemic classification types for the Mycelix ecosystem.
//!
//! Shared between:
//! - Prism browser (`plexus/plexus-common`) — search + display
//! - Knowledge cluster (`knowledge-leptos-types`) — Leptos frontend
//! - Knowledge zomes (`claims/integrity`) — Holochain DHT
//! - Mock data (`mycelix-mock-data`) — offline fallback
//!
//! The LEM Cube: Empirical × Normative × Materiality
//! Each dimension is discrete (0-4, 0-3, 0-3) with clear semantics.

use serde::{Deserialize, Serialize};

// ============================================================================
// Empirical Level (E0-E4) — How verified is this claim?
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EmpiricalLevel {
    E0, // Unverified — no empirical evidence
    E1, // Preliminary — initial observations or anecdotal
    E2, // Tested — systematic study, single source
    E3, // Replicated — multiple independent confirmations
    E4, // Established — scientific consensus, reproducible
}

impl EmpiricalLevel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::E0 => "Unverified",
            Self::E1 => "Preliminary",
            Self::E2 => "Tested",
            Self::E3 => "Replicated",
            Self::E4 => "Established",
        }
    }

    /// Convert to continuous 0.0-1.0 scale.
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::E0 => 0.0,
            Self::E1 => 0.25,
            Self::E2 => 0.5,
            Self::E3 => 0.75,
            Self::E4 => 1.0,
        }
    }

    pub fn from_f32(v: f32) -> Self {
        if v >= 0.875 {
            Self::E4
        } else if v >= 0.625 {
            Self::E3
        } else if v >= 0.375 {
            Self::E2
        } else if v >= 0.125 {
            Self::E1
        } else {
            Self::E0
        }
    }
}

// ============================================================================
// Normative Level (N0-N3) — What social consensus exists?
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NormativeLevel {
    N0, // Raw — no normative evaluation
    N1, // Contested — active disagreement
    N2, // Emerging — growing consensus
    N3, // Endorsed — broad agreement
}

impl NormativeLevel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::N0 => "Raw",
            Self::N1 => "Contested",
            Self::N2 => "Emerging",
            Self::N3 => "Endorsed",
        }
    }

    pub fn as_f32(&self) -> f32 {
        match self {
            Self::N0 => 0.0,
            Self::N1 => 0.33,
            Self::N2 => 0.67,
            Self::N3 => 1.0,
        }
    }
}

// ============================================================================
// Materiality Level (M0-M3) — How impactful is this knowledge?
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MaterialityLevel {
    M0, // Abstract — purely theoretical
    M1, // Potential — theoretical implications
    M2, // Applicable — practical applications
    M3, // Transformative — significant real-world impact
}

impl MaterialityLevel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::M0 => "Abstract",
            Self::M1 => "Potential",
            Self::M2 => "Applicable",
            Self::M3 => "Transformative",
        }
    }

    pub fn as_f32(&self) -> f32 {
        match self {
            Self::M0 => 0.0,
            Self::M1 => 0.33,
            Self::M2 => 0.67,
            Self::M3 => 1.0,
        }
    }
}

// ============================================================================
// Claim Type
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimType {
    Fact,
    Opinion,
    Prediction,
    Hypothesis,
    Definition,
    Historical,
    Normative,
    Narrative,
}

impl ClaimType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fact => "Fact",
            Self::Opinion => "Opinion",
            Self::Prediction => "Prediction",
            Self::Hypothesis => "Hypothesis",
            Self::Definition => "Definition",
            Self::Historical => "Historical",
            Self::Normative => "Normative",
            Self::Narrative => "Narrative",
        }
    }

    pub fn all() -> &'static [ClaimType] {
        &[
            Self::Fact,
            Self::Opinion,
            Self::Prediction,
            Self::Hypothesis,
            Self::Definition,
            Self::Historical,
            Self::Normative,
            Self::Narrative,
        ]
    }
}

// ============================================================================
// Unified Epistemic Classification (the LEM Cube coordinate)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpistemicClassification {
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
    pub confidence: f64,
}

impl EpistemicClassification {
    /// Composite score: weighted average of all three dimensions.
    pub fn composite_score(&self) -> f64 {
        let e = self.empirical.as_f32() as f64;
        let n = self.normative.as_f32() as f64;
        let m = self.materiality.as_f32() as f64;
        (e * 0.5 + n * 0.25 + m * 0.25) * self.confidence
    }
}

// ============================================================================
// Unified Claim — the canonical claim representation
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedClaim {
    pub id: String,
    pub content: String,
    pub classification: EpistemicClassification,
    pub claim_type: ClaimType,
    pub author: String,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub created: i64,
    pub version: u32,
}
