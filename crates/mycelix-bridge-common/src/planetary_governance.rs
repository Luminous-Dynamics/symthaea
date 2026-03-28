// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Planetary Governance: interplanetary voting, latency penalties, federation.
//!
//! Extends Mycelix governance with physics-aware constraints. Communication
//! latency (4-90 minutes one-way) fundamentally changes how democracy works
//! across planets. This module defines the latency penalty model and the
//! consciousness mapping between the multiworld sim's 6D vector and
//! Mycelix's 4D ConsciousnessProfile.
//!
//! # Layer 2: Consciousness Mapping (6D ↔ 4D)
//!
//! The multiworld sim tracks 6 consciousness dimensions per agent.
//! Mycelix production uses 4 dimensions. This module provides the canonical
//! mapping between them.
//!
//! # Layer 4: Governance Parameter Extraction
//!
//! The multiworld sim's 1000-year stress tests validate Mycelix governance
//! thresholds. This module defines the interface for extracting validated
//! parameters.

use crate::earth_colony_protocol::PlanetaryBody;
use serde::{Deserialize, Serialize};

// ============================================================================
// Layer 2: Consciousness Mapping (6D Sim ↔ 4D Mycelix)
// ============================================================================

/// The multiworld sim's 6-dimensional consciousness vector.
/// Maps to Mycelix's 4D ConsciousnessProfile via `to_mycelix_profile()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimConsciousnessVector {
    /// Overall consciousness level [0, 1].
    pub level: f64,
    /// Meta-awareness capacity [0, 1].
    pub meta_awareness: f64,
    /// Internal coherence [0, 1].
    pub coherence: f64,
    /// Care/compassion activation [0, 1].
    pub care_activation: f64,
    /// Alignment with Eight Harmonies [0, 1].
    pub harmonic_alignment: f64,
    /// Epistemic confidence calibration [0, 1].
    pub epistemic_confidence: f64,
}

/// Mycelix's 4-dimensional consciousness profile (mirrors ConsciousnessProfile).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MycelixConsciousnessMapping {
    /// Identity: calibrated self-knowledge. From level + meta + epistemic.
    pub identity: f64,
    /// Reputation: internal consistency over time. From coherence.
    pub reputation: f64,
    /// Community: mutual aid capacity. From care_activation.
    pub community: f64,
    /// Engagement: cultural participation. From harmonic_alignment.
    pub engagement: f64,
    /// Combined score (25% identity + 25% reputation + 30% community + 20% engagement).
    pub combined: f64,
}

impl SimConsciousnessVector {
    /// Map the sim's 6D consciousness → Mycelix's 4D profile.
    ///
    /// This is the canonical bridge between the testing ground and production.
    /// The mapping preserves the semantic intent of each Mycelix dimension:
    /// - Identity = "who you are" (self-knowledge + meta-awareness + epistemic calibration)
    /// - Reputation = "your consistency" (coherence over time)
    /// - Community = "your care for others" (care_activation)
    /// - Engagement = "your participation" (harmonic_alignment)
    pub fn to_mycelix_profile(&self) -> MycelixConsciousnessMapping {
        let identity = (self.level * 0.4 + self.meta_awareness * 0.3
            + self.epistemic_confidence * 0.3).clamp(0.0, 1.0);
        let reputation = self.coherence.clamp(0.0, 1.0);
        let community = self.care_activation.clamp(0.0, 1.0);
        let engagement = self.harmonic_alignment.clamp(0.0, 1.0);
        let combined = identity * 0.25 + reputation * 0.25
            + community * 0.30 + engagement * 0.20;
        MycelixConsciousnessMapping { identity, reputation, community, engagement, combined }
    }
}

// ============================================================================
// Layer 4: Communication Latency Governance
// ============================================================================

/// Communication latency class for governance participation.
///
/// Determines how planetary distance affects voting coherence and
/// governance participation quality. At >30 min one-way latency,
/// real-time deliberation is impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyClass {
    /// <2s one-way (Earth-Moon). Real-time possible.
    RealTime,
    /// 3-22 min one-way (Earth-Mars). Near-real-time, async preferred.
    NearRealTime,
    /// 33-54 min one-way (Earth-Jupiter). Async only.
    HighLatency,
    /// 67-90 min one-way (Earth-Saturn). Severely async.
    ExtremeLatency,
}

impl LatencyClass {
    pub fn from_body(body: PlanetaryBody) -> Self {
        match body {
            PlanetaryBody::Earth | PlanetaryBody::Moon | PlanetaryBody::OrbitalHabitat => Self::RealTime,
            PlanetaryBody::Mars | PlanetaryBody::Ceres => Self::NearRealTime,
            PlanetaryBody::Europa => Self::HighLatency,
            PlanetaryBody::Titan => Self::ExtremeLatency,
        }
    }

    /// Governance coherence penalty [0.0, 1.0].
    /// 0.0 = no penalty (real-time), 1.0 = severely degraded.
    /// Represents the loss of deliberative quality from communication delay.
    pub fn governance_penalty(&self) -> f64 {
        match self {
            Self::RealTime => 0.0,
            Self::NearRealTime => 0.15,      // Can't do live debate, but async works
            Self::HighLatency => 0.35,        // Deliberation takes days per round
            Self::ExtremeLatency => 0.50,     // Effectively autonomous governance
        }
    }

    /// Vote weight multiplier. Distant colonies' votes are discounted
    /// in time-sensitive decisions (not in constitutional matters).
    pub fn vote_weight_multiplier(&self) -> f64 {
        1.0 - self.governance_penalty() * 0.5
    }
}

/// Interplanetary vote with latency-aware processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterplanetaryVote {
    /// The voter's planetary body.
    pub voter_body: PlanetaryBody,
    /// The voter's consciousness tier (from Mycelix profile).
    pub voter_tier: u8,
    /// Vote payload (proposal hash + yes/no/abstain).
    pub vote: VoteChoice,
    /// Tick when vote was cast (at voter's location).
    pub cast_tick: u32,
    /// Tick when vote arrives at the counting body (cast_tick + light_delay_ticks).
    pub arrival_tick: u32,
    /// Whether this is a time-sensitive vote (latency penalty applies).
    pub time_sensitive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteChoice {
    Yes,
    No,
    Abstain,
}

// ============================================================================
// Layer 3: TEND ↔ Cobb-Douglas Bridge
// ============================================================================

/// Maps Cobb-Douglas sector output to TEND service hours.
///
/// The multiworld sim uses 8-sector Cobb-Douglas production.
/// Mycelix uses TEND mutual credit (1 TEND = 1 hour of labor).
/// This bridge converts between the two economic models.
///
/// Sector indices: 0=engineering, 1=agriculture, 2=medicine,
/// 3=governance, 4=science, 5=education, 6=art_culture, 7=logistics
pub fn cobb_douglas_to_tend_hours(sector_output: &[f64; 8], population: usize) -> TendEquivalence {
    let pop = population.max(1) as f64;
    // Each unit of Cobb-Douglas output ≈ 10 person-hours of labor
    // (calibrated to match TEND's 1:1 hour accounting)
    let hours_per_unit = 10.0;

    TendEquivalence {
        tech_support_hours: sector_output[0] * hours_per_unit / pop,
        food_services_hours: sector_output[1] * hours_per_unit / pop,
        care_work_hours: sector_output[2] * hours_per_unit / pop,
        governance_hours: sector_output[3] * hours_per_unit / pop,
        research_hours: sector_output[4] * hours_per_unit / pop,
        education_hours: sector_output[5] * hours_per_unit / pop,
        creative_hours: sector_output[6] * hours_per_unit / pop,
        logistics_hours: sector_output[7] * hours_per_unit / pop,
        total_per_capita: sector_output.iter().sum::<f64>() * hours_per_unit / pop,
    }
}

/// TEND-equivalent hours per capita per tick.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TendEquivalence {
    pub tech_support_hours: f64,
    pub food_services_hours: f64,
    pub care_work_hours: f64,
    pub governance_hours: f64,
    pub research_hours: f64,
    pub education_hours: f64,
    pub creative_hours: f64,
    pub logistics_hours: f64,
    /// Total TEND hours per capita per tick.
    pub total_per_capita: f64,
}

impl TendEquivalence {
    /// Check if TEND balance limits are survivable.
    /// Returns true if per-capita hours are within ±40 TEND (normal limit).
    pub fn within_tend_limits(&self) -> bool {
        self.total_per_capita <= 40.0
    }

    /// Vitality tier recommendation based on output level.
    pub fn recommended_vitality_tier(&self) -> &'static str {
        if self.total_per_capita > 30.0 { "Emergency" }
        else if self.total_per_capita > 20.0 { "High" }
        else if self.total_per_capita > 10.0 { "Elevated" }
        else { "Normal" }
    }
}

// ============================================================================
// Layer 4: Governance Parameter Extraction
// ============================================================================

/// Validated governance parameters from multiworld sim stress testing.
///
/// These values should be extracted from multi-seed 1000-year sim runs
/// and fed back into `consciousness_thresholds.rs` as production defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidatedGovernanceParams {
    /// Tested veto override threshold (fraction). Default: 0.80.
    pub veto_override_threshold: f64,
    /// Tested maximum emergency sessions before cooldown. Default: 3.
    pub max_emergency_sessions: u32,
    /// Year at which Guardian capture first occurs (if ever).
    pub guardian_capture_year: Option<f64>,
    /// Minimum colony size for stable self-governance.
    pub min_self_governance_population: u32,
    /// Consciousness gating threshold below which governance collapses.
    pub min_viable_phi: f64,
    /// Constitutional amendment rate that prevents calcification.
    pub healthy_amendment_rate: f64,
    /// Number of seeds tested.
    pub seeds_tested: u32,
    /// Mean CVS across seeds.
    pub mean_cvs: f64,
    /// Whether civilization survived in all seeds.
    pub all_survived: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_mapping_preserves_range() {
        let sim = SimConsciousnessVector {
            level: 0.8, meta_awareness: 0.6, coherence: 0.7,
            care_activation: 0.9, harmonic_alignment: 0.5, epistemic_confidence: 0.4,
        };
        let mycelix = sim.to_mycelix_profile();
        assert!(mycelix.identity >= 0.0 && mycelix.identity <= 1.0);
        assert!(mycelix.combined >= 0.0 && mycelix.combined <= 1.0);
        // High care → high community
        assert!(mycelix.community > 0.8);
    }

    #[test]
    fn test_latency_classes() {
        assert_eq!(LatencyClass::from_body(PlanetaryBody::Moon), LatencyClass::RealTime);
        assert_eq!(LatencyClass::from_body(PlanetaryBody::Mars), LatencyClass::NearRealTime);
        assert_eq!(LatencyClass::from_body(PlanetaryBody::Europa), LatencyClass::HighLatency);
        assert_eq!(LatencyClass::from_body(PlanetaryBody::Titan), LatencyClass::ExtremeLatency);
    }

    #[test]
    fn test_tend_bridge() {
        let output = [10.0, 15.0, 8.0, 5.0, 12.0, 10.0, 3.0, 7.0];
        let tend = cobb_douglas_to_tend_hours(&output, 100);
        assert!(tend.total_per_capita > 0.0);
        assert!(tend.food_services_hours > tend.logistics_hours);
    }

    #[test]
    fn test_governance_penalty_ordering() {
        assert!(LatencyClass::RealTime.governance_penalty()
            < LatencyClass::NearRealTime.governance_penalty());
        assert!(LatencyClass::NearRealTime.governance_penalty()
            < LatencyClass::HighLatency.governance_penalty());
        assert!(LatencyClass::HighLatency.governance_penalty()
            < LatencyClass::ExtremeLatency.governance_penalty());
    }
}
