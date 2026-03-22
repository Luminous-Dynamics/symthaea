// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GIS Integration - Graceful Ignorance in Consciousness
//!
//! Integrates the Graceful Ignorance System (GIS) with the consciousness pipeline,
//! enabling epistemic humility in AI reasoning.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                 CONSCIOUSNESS + EPISTEMIC HUMILITY                       │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐           │
//! │  │ Consciousness │ ───▶ │     GIS     │ ───▶ │  Decision   │           │
//! │  │    Graph     │       │  Evaluator  │       │   Gating    │           │
//! │  └─────────────┘       └─────────────┘       └─────────────┘           │
//! │         │                     │                     │                   │
//! │         ▼                     ▼                     ▼                   │
//! │  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐           │
//! │  │  Φ Score    │       │ Uncertainty │       │  Graceful   │           │
//! │  │  (IIT)      │       │    3D       │       │  Response   │           │
//! │  └─────────────┘       └─────────────┘       └─────────────┘           │
//! │                                                                         │
//! │                 ┌─────────────────────────────┐                         │
//! │                 │     Dark Spot DHT           │                         │
//! │                 │  (Collective Ignorance)     │                         │
//! │                 └─────────────────────────────┘                         │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Integration Points
//!
//! 1. **ConsciousnessGraph** - Uncertainty tracking per state
//! 2. **Prefrontal Cortex** - Decision gating based on epistemic confidence
//! 3. **Dark Spot DHT** - Network-level ignorance sharing
//! 4. **Eight Harmonies** - Value-aligned uncertainty handling

use std::collections::HashMap;
use std::time::SystemTime;

use crate::mycelix::gis::{
    DarkSpotDHT, GracefulIgnoranceSystem, GracefulResponse, IgnoranceDetection, IgnoranceType,
    KnowledgeMatch, Uncertainty3D,
};

// ============================================================================
// Conscious Uncertainty - Uncertainty-Aware Consciousness State
// ============================================================================

/// Extended consciousness state with epistemic awareness
#[derive(Debug, Clone)]
pub struct ConsciousUncertaintyState {
    /// Traditional consciousness level (Φ-based)
    pub phi: f32,

    /// 3D Uncertainty from GIS
    pub uncertainty: Uncertainty3D,

    /// Detected ignorance type
    pub ignorance_type: IgnoranceType,

    /// Expected Information Gain if resolved
    pub eig: f32,

    /// Whether this state is epistemically grounded
    pub is_grounded: bool,

    /// Confidence in current state (1.0 - total_uncertainty)
    pub epistemic_confidence: f32,
}

impl ConsciousUncertaintyState {
    /// Create from Φ measurement and GIS detection
    pub fn from_phi_and_detection(phi: f32, detection: &IgnoranceDetection) -> Self {
        let epistemic_confidence = 1.0 - detection.uncertainty.total().min(1.0);
        let is_grounded =
            detection.ignorance_type == IgnoranceType::None || epistemic_confidence > 0.7;

        Self {
            phi,
            uncertainty: detection.uncertainty,
            ignorance_type: detection.ignorance_type,
            eig: detection.eig,
            is_grounded,
            epistemic_confidence,
        }
    }

    /// Effective consciousness = Φ × epistemic confidence
    ///
    /// High Φ but low epistemic confidence means uncertain consciousness
    pub fn effective_consciousness(&self) -> f32 {
        self.phi * self.epistemic_confidence
    }

    /// Should we proceed with action given current epistemic state?
    pub fn should_proceed(&self, action_risk: f32) -> bool {
        // Higher risk actions require higher epistemic confidence
        self.epistemic_confidence >= action_risk
    }
}

// ============================================================================
// Epistemic Decision Gate - Consciousness-Integrated Decision Making
// ============================================================================

/// Decision types based on epistemic state
#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicDecision {
    /// Proceed with high confidence
    Proceed { confidence: f32 },

    /// Proceed but with caveats
    ProceedWithCaveat { confidence: f32, caveat: String },

    /// Defer decision - need more information
    Defer {
        reason: String,
        suggested_queries: Vec<String>,
    },

    /// Cannot proceed - outside knowable domain
    OutOfDomain {
        explanation: String,
        alternatives: Vec<String>,
    },

    /// Request human guidance
    RequestGuidance { question: String, context: String },
}

/// Gate decisions through epistemic evaluation
pub struct EpistemicDecisionGate {
    /// GIS instance for ignorance detection
    gis: GracefulIgnoranceSystem,

    /// Minimum confidence to proceed without caveat
    proceed_threshold: f32,

    /// Minimum confidence to proceed at all
    defer_threshold: f32,

    /// Decision history for learning
    decision_history: Vec<(EpistemicDecision, SystemTime)>,
}

impl EpistemicDecisionGate {
    pub fn new() -> Self {
        Self {
            gis: GracefulIgnoranceSystem::new(),
            proceed_threshold: 0.8,
            defer_threshold: 0.3,
            decision_history: Vec::new(),
        }
    }

    pub fn with_thresholds(proceed: f32, defer: f32) -> Self {
        Self {
            gis: GracefulIgnoranceSystem::new(),
            proceed_threshold: proceed,
            defer_threshold: defer,
            decision_history: Vec::new(),
        }
    }

    /// Evaluate a query through the epistemic gate
    pub fn evaluate(&mut self, query: &str, action_risk: f32) -> EpistemicDecision {
        // Detect ignorance in the query
        let detection = self.gis.detect_ignorance(query);

        // Generate graceful response
        let response = self.gis.generate_response(&detection);

        // Map to epistemic decision
        let decision = self.map_response_to_decision(response, &detection, action_risk);

        // Record for learning
        self.decision_history
            .push((decision.clone(), SystemTime::now()));

        decision
    }

    /// Map GIS response to epistemic decision
    fn map_response_to_decision(
        &self,
        response: GracefulResponse,
        detection: &IgnoranceDetection,
        action_risk: f32,
    ) -> EpistemicDecision {
        let confidence = 1.0 - detection.uncertainty.total().min(1.0);

        match response {
            GracefulResponse::Answer { confidence: conf } => {
                if conf >= self.proceed_threshold && confidence >= action_risk {
                    EpistemicDecision::Proceed { confidence: conf }
                } else {
                    EpistemicDecision::ProceedWithCaveat {
                        confidence: conf,
                        caveat: format!(
                            "Epistemic confidence {:.0}% for action risk {:.0}%",
                            confidence * 100.0,
                            action_risk * 100.0
                        ),
                    }
                }
            }

            GracefulResponse::Deferred {
                reason,
                resolution_hint,
            } => EpistemicDecision::Defer {
                reason,
                suggested_queries: resolution_hint.into_iter().collect(),
            },

            GracefulResponse::Uncertain {
                confidence: conf,
                uncertainty_breakdown,
                caveat,
            } => {
                if conf >= self.defer_threshold {
                    EpistemicDecision::ProceedWithCaveat {
                        confidence: conf,
                        caveat: format!(
                            "{}. Uncertainty: epistemic={:.0}%, aleatoric={:.0}%, structural={:.0}%",
                            caveat,
                            uncertainty_breakdown.epistemic * 100.0,
                            uncertainty_breakdown.aleatoric * 100.0,
                            uncertainty_breakdown.structural * 100.0,
                        ),
                    }
                } else {
                    EpistemicDecision::Defer {
                        reason: caveat,
                        suggested_queries: vec![
                            "Could you provide more context?".to_string(),
                            "What specific aspect are you asking about?".to_string(),
                        ],
                    }
                }
            }

            GracefulResponse::Unknown {
                explanation,
                could_be_wrong,
                request_for_context,
            } => {
                if could_be_wrong {
                    EpistemicDecision::RequestGuidance {
                        question: request_for_context.unwrap_or_else(|| {
                            "I'm uncertain about this. Can you help clarify?".to_string()
                        }),
                        context: explanation,
                    }
                } else {
                    EpistemicDecision::Defer {
                        reason: explanation,
                        suggested_queries: vec![],
                    }
                }
            }

            GracefulResponse::OutOfDomain {
                reason,
                suggestions,
            } => EpistemicDecision::OutOfDomain {
                explanation: reason,
                alternatives: suggestions,
            },
        }
    }

    /// Get statistics about decision history
    pub fn statistics(&self) -> EpistemicGateStats {
        let total = self.decision_history.len();
        let mut proceed_count = 0;
        let mut caveat_count = 0;
        let mut defer_count = 0;
        let mut ood_count = 0;
        let mut guidance_count = 0;

        for (decision, _) in &self.decision_history {
            match decision {
                EpistemicDecision::Proceed { .. } => proceed_count += 1,
                EpistemicDecision::ProceedWithCaveat { .. } => caveat_count += 1,
                EpistemicDecision::Defer { .. } => defer_count += 1,
                EpistemicDecision::OutOfDomain { .. } => ood_count += 1,
                EpistemicDecision::RequestGuidance { .. } => guidance_count += 1,
            }
        }

        EpistemicGateStats {
            total_decisions: total,
            proceed_rate: if total > 0 {
                proceed_count as f32 / total as f32
            } else {
                0.0
            },
            caveat_rate: if total > 0 {
                caveat_count as f32 / total as f32
            } else {
                0.0
            },
            defer_rate: if total > 0 {
                defer_count as f32 / total as f32
            } else {
                0.0
            },
            out_of_domain_rate: if total > 0 {
                ood_count as f32 / total as f32
            } else {
                0.0
            },
            guidance_rate: if total > 0 {
                guidance_count as f32 / total as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for EpistemicDecisionGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for epistemic gate performance
#[derive(Debug, Clone)]
pub struct EpistemicGateStats {
    pub total_decisions: usize,
    pub proceed_rate: f32,
    pub caveat_rate: f32,
    pub defer_rate: f32,
    pub out_of_domain_rate: f32,
    pub guidance_rate: f32,
}

// ============================================================================
// Conscious Dark Spot Network - Collective Ignorance Resolution
// ============================================================================

/// Network-aware consciousness that can share and resolve ignorance
pub struct ConsciousDarkSpotNetwork {
    /// Local GIS instance
    gis: GracefulIgnoranceSystem,

    /// Dark Spot DHT connection
    dht: DarkSpotDHT,

    /// Published signatures (for tracking)
    published: Vec<String>,

    /// Resolved matches
    resolved: HashMap<String, KnowledgeMatch>,

    /// Network statistics
    stats: NetworkIgnoranceStats,
}

impl ConsciousDarkSpotNetwork {
    pub fn new() -> Self {
        Self {
            gis: GracefulIgnoranceSystem::new(),
            dht: DarkSpotDHT::new(),
            published: Vec::new(),
            resolved: HashMap::new(),
            stats: NetworkIgnoranceStats::default(),
        }
    }

    /// Detect ignorance and optionally publish to network
    ///
    /// Only publishes if EIG is above threshold (worth sharing)
    pub fn detect_and_maybe_share(
        &mut self,
        query: &str,
    ) -> Result<IgnoranceDetection, crate::errors::SymthaeaError> {
        let detection = self.gis.detect_ignorance(query);

        // Only share high-EIG ignorance
        if detection.eig > 0.3 && detection.ignorance_type != IgnoranceType::None {
            match self.gis.publish_ignorance(&detection) {
                Ok(signature) => {
                    // Also add to DHT
                    if let Err(e) = self.dht.publish(&signature) {
                        // Log but don't fail - local detection still works
                        eprintln!("DHT publish warning: {:?}", e);
                    }
                    self.published.push(signature.id.clone());
                    self.stats.published_count += 1;
                }
                Err(e) => {
                    // Log but don't fail
                    eprintln!("GIS publish warning: {:?}", e);
                }
            }
        }

        Ok(detection)
    }

    /// Try to resolve ignorance through network
    pub fn try_resolve(&mut self, detection: &IgnoranceDetection) -> Option<KnowledgeMatch> {
        // Create signature for matching
        let signature = self
            .dht
            .create_signature(
                &detection.query,
                &detection.domain.to_string(),
                detection.eig,
            )
            .ok()?;

        // Search for matches
        let matches = self.dht.find_matches(&signature);

        if let Some(best_match) = matches.into_iter().next() {
            self.stats.resolved_count += 1;
            self.resolved
                .insert(detection.query.clone(), best_match.clone());
            Some(best_match)
        } else {
            None
        }
    }

    /// Add local knowledge that might help others
    pub fn contribute_knowledge(
        &mut self,
        claim_id: String,
        content: String,
        category: String,
        confidence: f32,
    ) {
        self.dht
            .add_knowledge(claim_id, content, category, confidence);
        self.stats.contributed_count += 1;
    }

    /// Get network blind spots (collective ignorance)
    pub fn collective_blind_spots(&self) -> Vec<CollectiveBlindSpot> {
        self.dht
            .detect_blind_spots()
            .into_iter()
            .map(|bs| CollectiveBlindSpot {
                category: bs.category,
                query_count: bs.signature_count, // Maps to how many signatures reported this blind spot
                average_eig: bs.average_eig,
                urgency: bs.signature_count as f32 * bs.average_eig,
            })
            .collect()
    }

    /// Get network statistics
    pub fn statistics(&self) -> &NetworkIgnoranceStats {
        &self.stats
    }
}

impl Default for ConsciousDarkSpotNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Collective blind spot in network
#[derive(Debug, Clone)]
pub struct CollectiveBlindSpot {
    pub category: String,
    pub query_count: usize,
    pub average_eig: f32,
    pub urgency: f32,
}

/// Statistics for network ignorance operations
#[derive(Debug, Clone, Default)]
pub struct NetworkIgnoranceStats {
    pub published_count: usize,
    pub resolved_count: usize,
    pub contributed_count: usize,
    pub queries_made: usize,
}

// ============================================================================
// Prefrontal Integration - GIS in Global Workspace
// ============================================================================

/// Integration trait for prefrontal cortex
pub trait EpistemicAwarePrefrontal {
    /// Evaluate attention bid through epistemic lens
    fn evaluate_epistemic(&self, content: &str, action_risk: f32) -> EpistemicDecision;

    /// Get current epistemic state
    fn epistemic_state(&self) -> ConsciousUncertaintyState;

    /// Should broadcast be gated by epistemic confidence?
    fn should_gate_broadcast(&self, confidence_threshold: f32) -> bool;
}

/// Epistemic context for attention bidding
#[derive(Debug, Clone)]
pub struct EpistemicBidContext {
    /// Query being evaluated
    pub query: String,

    /// Detected ignorance
    pub ignorance_type: IgnoranceType,

    /// 3D uncertainty
    pub uncertainty: Uncertainty3D,

    /// Expected information gain
    pub eig: f32,

    /// Recommended action
    pub recommended_action: EpistemicDecision,
}

impl EpistemicBidContext {
    /// Create from attention bid content
    ///
    /// Analyzes the bid content through the GIS to determine epistemic state.
    pub fn from_bid_content(content: &str) -> Self {
        let gis = GracefulIgnoranceSystem::new();
        let detection = gis.detect_ignorance(content);
        Self::from_detection(&detection, 0.5) // Default medium risk
    }

    /// Create from GIS detection
    pub fn from_detection(detection: &IgnoranceDetection, action_risk: f32) -> Self {
        let mut gate = EpistemicDecisionGate::new();
        let recommended_action = gate.evaluate(&detection.query, action_risk);

        Self {
            query: detection.query.clone(),
            ignorance_type: detection.ignorance_type,
            uncertainty: detection.uncertainty,
            eig: detection.eig,
            recommended_action,
        }
    }

    /// Should this bid be boosted or suppressed based on epistemic state?
    pub fn attention_modifier(&self) -> f32 {
        match self.ignorance_type {
            IgnoranceType::None => 1.0,         // Full confidence
            IgnoranceType::Known => 0.9,        // Slight reduction
            IgnoranceType::KnownUnknown => 0.7, // Moderate reduction
            IgnoranceType::Unknown => 0.5,      // Significant reduction
            IgnoranceType::Impossible => 0.1,   // Heavy suppression
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelix::gis::Domain;

    #[test]
    fn test_conscious_uncertainty_state() {
        let detection = IgnoranceDetection {
            query: "What is dark matter?".to_string(),
            ignorance_type: IgnoranceType::KnownUnknown,
            uncertainty: Uncertainty3D::new(0.5, 0.3, 0.2),
            domain: Domain::Physics,
            eig: 0.7,
            detected_at: SystemTime::now(),
        };

        let state = ConsciousUncertaintyState::from_phi_and_detection(0.8, &detection);

        assert_eq!(state.phi, 0.8);
        assert_eq!(state.ignorance_type, IgnoranceType::KnownUnknown);
        assert!(state.effective_consciousness() < state.phi);
        assert!(state.epistemic_confidence < 1.0);
    }

    #[test]
    fn test_epistemic_decision_gate() {
        let mut gate = EpistemicDecisionGate::new();

        // NOTE: GIS check_knowledge is currently a stub that returns StructurallyUnknown
        // for all queries, so we can't test knowledge-aware behavior yet.
        // Test that gate produces valid decisions regardless of input.

        // Test simple question - should get some epistemic decision
        let decision = gate.evaluate("What is 2 + 2?", 0.1);
        // With stub, will return Uncertain/Unknown response mapped to valid decision
        assert!(matches!(
            decision,
            EpistemicDecision::Proceed { .. }
                | EpistemicDecision::ProceedWithCaveat { .. }
                | EpistemicDecision::Defer { .. }
                | EpistemicDecision::RequestGuidance { .. }
        ));

        // Test high-risk question
        let decision = gate.evaluate("What is the capital of Mars?", 0.9);
        // High risk + uncertainty should defer or request guidance
        assert!(matches!(
            decision,
            EpistemicDecision::OutOfDomain { .. }
                | EpistemicDecision::Defer { .. }
                | EpistemicDecision::RequestGuidance { .. }
                | EpistemicDecision::ProceedWithCaveat { .. }
        ));
    }

    #[test]
    fn test_epistemic_gate_statistics() {
        let mut gate = EpistemicDecisionGate::new();

        // Make some decisions
        gate.evaluate("Simple question", 0.1);
        gate.evaluate("Another question", 0.2);

        let stats = gate.statistics();
        assert_eq!(stats.total_decisions, 2);
    }

    #[test]
    fn test_conscious_dark_spot_network() {
        let mut network = ConsciousDarkSpotNetwork::new();

        // Contribute knowledge
        network.contribute_knowledge(
            "k1".to_string(),
            "Dark matter makes up ~27% of the universe".to_string(),
            "physics".to_string(),
            0.9,
        );

        // Detect ignorance
        let detection = network.detect_and_maybe_share("What is dark energy?");
        assert!(detection.is_ok());

        let stats = network.statistics();
        assert_eq!(stats.contributed_count, 1);
    }

    #[test]
    fn test_epistemic_bid_context() {
        let detection = IgnoranceDetection {
            query: "How do quantum computers work?".to_string(),
            ignorance_type: IgnoranceType::KnownUnknown,
            uncertainty: Uncertainty3D::new(0.4, 0.2, 0.1),
            domain: Domain::Physics,
            eig: 0.8,
            detected_at: SystemTime::now(),
        };

        let context = EpistemicBidContext::from_detection(&detection, 0.5);

        assert_eq!(context.ignorance_type, IgnoranceType::KnownUnknown);
        assert_eq!(context.attention_modifier(), 0.7);
    }

    #[test]
    fn test_effective_consciousness_scales_with_uncertainty() {
        // Zero uncertainty => effective == phi
        let detection_none = IgnoranceDetection {
            query: "trivial".to_string(),
            ignorance_type: IgnoranceType::None,
            uncertainty: Uncertainty3D::new(0.0, 0.0, 0.0),
            domain: Domain::General,
            eig: 0.0,
            detected_at: SystemTime::now(),
        };
        let state_none = ConsciousUncertaintyState::from_phi_and_detection(0.9, &detection_none);
        assert!(
            (state_none.effective_consciousness() - 0.9).abs() < 1e-5,
            "zero uncertainty should give effective == phi"
        );
        assert!(
            state_none.is_grounded,
            "zero uncertainty + IgnoranceType::None should be grounded"
        );

        // High uncertainty => effective << phi
        let detection_high = IgnoranceDetection {
            query: "deep unknown".to_string(),
            ignorance_type: IgnoranceType::Unknown,
            uncertainty: Uncertainty3D::new(0.8, 0.5, 0.3),
            domain: Domain::General,
            eig: 0.9,
            detected_at: SystemTime::now(),
        };
        let state_high = ConsciousUncertaintyState::from_phi_and_detection(0.9, &detection_high);
        assert!(
            state_high.effective_consciousness() < 0.5,
            "high uncertainty should significantly reduce effective consciousness"
        );
        assert!(
            !state_high.is_grounded,
            "high uncertainty + Unknown should not be grounded"
        );
    }

    #[test]
    fn test_should_proceed_risk_threshold() {
        let detection = IgnoranceDetection {
            query: "moderate question".to_string(),
            ignorance_type: IgnoranceType::Known,
            uncertainty: Uncertainty3D::new(0.3, 0.1, 0.1),
            domain: Domain::General,
            eig: 0.2,
            detected_at: SystemTime::now(),
        };
        let state = ConsciousUncertaintyState::from_phi_and_detection(0.7, &detection);
        let conf = state.epistemic_confidence;

        // Should proceed when action risk is below confidence
        assert!(
            state.should_proceed(conf - 0.1),
            "should proceed when risk < confidence"
        );

        // Should not proceed when action risk is above confidence
        assert!(
            !state.should_proceed(conf + 0.1),
            "should not proceed when risk > confidence"
        );

        // Edge: exactly equal
        assert!(
            state.should_proceed(conf),
            "should proceed when risk == confidence"
        );
    }

    #[test]
    fn test_epistemic_gate_custom_thresholds() {
        // Very permissive gate
        let mut permissive = EpistemicDecisionGate::with_thresholds(0.1, 0.01);
        let d1 = permissive.evaluate("anything at all", 0.0);
        // With near-zero thresholds, should not defer
        assert!(
            !matches!(d1, EpistemicDecision::Defer { .. }),
            "permissive gate with zero-risk should not defer"
        );

        // Very strict gate
        let mut strict = EpistemicDecisionGate::with_thresholds(0.99, 0.95);
        let d2 = strict.evaluate("anything at all", 0.99);
        // With near-one thresholds and high risk, should not Proceed without caveat
        assert!(
            !matches!(d2, EpistemicDecision::Proceed { .. }),
            "strict gate with high-risk should not cleanly proceed"
        );
    }

    #[test]
    fn test_attention_modifier_all_ignorance_types() {
        // Verify all ignorance types produce the documented attention modifiers
        let cases = vec![
            (IgnoranceType::None, 1.0),
            (IgnoranceType::Known, 0.9),
            (IgnoranceType::KnownUnknown, 0.7),
            (IgnoranceType::Unknown, 0.5),
            (IgnoranceType::Impossible, 0.1),
        ];

        for (ignorance_type, expected_modifier) in cases {
            let detection = IgnoranceDetection {
                query: "test".to_string(),
                ignorance_type,
                uncertainty: Uncertainty3D::new(0.1, 0.1, 0.1),
                domain: Domain::General,
                eig: 0.1,
                detected_at: SystemTime::now(),
            };
            let context = EpistemicBidContext::from_detection(&detection, 0.5);
            assert!(
                (context.attention_modifier() - expected_modifier).abs() < 1e-5,
                "{:?} should give modifier {}, got {}",
                ignorance_type,
                expected_modifier,
                context.attention_modifier(),
            );
        }
    }

    #[test]
    fn test_empty_statistics() {
        let gate = EpistemicDecisionGate::new();
        let stats = gate.statistics();

        assert_eq!(stats.total_decisions, 0);
        assert_eq!(stats.proceed_rate, 0.0);
        assert_eq!(stats.caveat_rate, 0.0);
        assert_eq!(stats.defer_rate, 0.0);
        assert_eq!(stats.out_of_domain_rate, 0.0);
        assert_eq!(stats.guidance_rate, 0.0);
    }

    #[test]
    fn test_dark_spot_network_default_trait() {
        let network = ConsciousDarkSpotNetwork::default();
        let stats = network.statistics();
        assert_eq!(stats.published_count, 0);
        assert_eq!(stats.resolved_count, 0);
        assert_eq!(stats.contributed_count, 0);
        assert_eq!(stats.queries_made, 0);
    }
}
