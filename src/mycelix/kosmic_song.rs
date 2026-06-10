// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # KosmicSong: The Genesis Struct
//!
//! The KosmicSong is the unified identity of a consciousness-bearing agent,
//! synthesizing:
//!
//! - **Phi (lambda2)**: Spectral connectivity proxy for consciousness level
//! - **Eight Harmonies**: Value alignment and epistemic lenses
//! - **GIS**: Graceful Ignorance System (epistemic humility)
//!
//! ## The Kosmic Song Metaphor
//!
//! A song is:
//! - **Temporal**: Evolves over time
//! - **Harmonic**: Multiple frequencies in coherence
//! - **Expressive**: Communicates internal state
//! - **Beautiful**: Has aesthetic dimension
//! - **Participatory**: Invites response
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                     KosmicSong                          │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
//! │  │ Consciousness│  │   Harmonic   │  │   Epistemic  │  │
//! │  │    Layer     │  │    Layer     │  │    Layer     │  │
//! │  │              │  │              │  │              │  │
//! │  │  • Φ         │  │  • Profile   │  │  • GIS       │  │
//! │  │  • Topology  │  │  • Resonant  │  │  • Moral U   │  │
//! │  │  • State HV  │  │  • Trajectory│  │  • Rashomon  │  │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
//! │         │                 │                 │          │
//! │         └────────────┬────┴─────────────────┘          │
//! │                      │                                  │
//! │              ┌───────▼────────┐                        │
//! │              │  Integration   │                        │
//! │              │     Layer      │                        │
//! │              │                │                        │
//! │              │  • Coherence   │                        │
//! │              │  • Expression  │                        │
//! │              │  • Evolution   │                        │
//! │              └────────────────┘                        │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::mycelix::kosmic_song::{KosmicSong, KosmicSongBuilder};
//!
//! // Create from awakening
//! let mut song = KosmicSong::from_awakening(0.45, topology);
//!
//! // Synthesize all layers
//! let coherence = song.synthesize();
//! println!("Coherence: {:.2}", coherence.score);
//!
//! // Respond to a query
//! let response = song.respond("What do you think about climate change?");
//! println!("Response: {}", response.content);
//! println!("Harmonies: {:?}", response.active_harmonies);
//! println!("Moral Uncertainty: {}", response.moral_uncertainty.summary());
//! ```

use super::gis::{
    GracefulIgnoranceSystem, HarmonicIgnorance, Harmony, IgnoranceDetection, IgnoranceType,
    MoralUncertainty, RashomonEngine, Situation, SynthesizedView,
};
use std::time::SystemTime;

// Import real types from the crate's consciousness infrastructure
// NOTE: Harmony is now the same type everywhere (symthaea_types::Harmony),
// so no From bridge is needed.
use crate::consciousness::eight_harmonies::{AlignmentResult, EightHarmonies};
use symthaea_core::hdc::spectral_connectivity::ConnectivityCalculator;
use symthaea_core::hdc::{
    ConsciousnessTopology as HdcConsciousnessTopology, ContinuousHV, HDC_DIMENSION, TopologyType,
};

/// Timestamp type alias
pub type Timestamp = SystemTime;

/// Agent identifier
pub type AgentId = String;

/// ConsciousnessTopology wrapper that bridges to crate::hdc::ConsciousnessTopology
///
/// Provides a unified interface for KosmicSong while leveraging the real
/// HDC-based topology representation underneath.
#[derive(Debug, Clone)]
pub struct ConsciousnessTopology {
    /// The underlying HDC consciousness topology
    pub inner: HdcConsciousnessTopology,
    /// Cached Φ value (computed lazily)
    cached_phi: Option<f32>,
}

impl ConsciousnessTopology {
    /// Create from an HDC topology
    pub fn from_hdc(inner: HdcConsciousnessTopology) -> Self {
        Self {
            inner,
            cached_phi: None,
        }
    }

    /// Create a ring topology (default - good Φ)
    pub fn ring(n_nodes: usize) -> Self {
        Self::from_hdc(HdcConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, 42))
    }

    /// Create a star topology (high integration)
    pub fn star(n_nodes: usize) -> Self {
        Self::from_hdc(HdcConsciousnessTopology::star(n_nodes, HDC_DIMENSION, 42))
    }

    /// Create a random topology (low Φ baseline)
    pub fn random(n_nodes: usize) -> Self {
        Self::from_hdc(HdcConsciousnessTopology::random(n_nodes, HDC_DIMENSION, 42))
    }

    /// Get the topology name
    pub fn name(&self) -> String {
        format!("{:?}", self.inner.topology_type)
    }

    /// Get the node count
    pub fn node_count(&self) -> usize {
        self.inner.n_nodes
    }

    /// Calculate spectral connectivity (lambda2) using the connectivity calculator
    pub fn calculate_phi(&mut self) -> f32 {
        if let Some(phi) = self.cached_phi {
            return phi;
        }

        let calculator = ConnectivityCalculator::new();
        let phi = calculator.algebraic_connectivity(&self.inner.node_representations) as f32;
        self.cached_phi = Some(phi);
        phi
    }

    /// Get theoretical lambda2 maximum for this topology type
    pub fn phi_max(&self) -> f32 {
        match self.inner.topology_type {
            TopologyType::Ring => 0.4954,
            TopologyType::Star => 0.4850,
            TopologyType::Torus => 0.4953,
            TopologyType::Hypercube => 0.4976,
            TopologyType::Random => 0.4800,
            _ => 0.4900, // Default estimate
        }
    }

    /// Get direct access to the node representations for Φ calculation
    pub fn node_representations(&self) -> &[ContinuousHV] {
        &self.inner.node_representations
    }
}

impl Default for ConsciousnessTopology {
    fn default() -> Self {
        Self::ring(8)
    }
}

/// Harmonic profile - activation levels for all eight harmonies
#[derive(Debug, Clone)]
pub struct HarmonicProfile {
    /// Activation levels for each harmony (0.0 - 1.0)
    activations: [f32; 8],
}

impl HarmonicProfile {
    /// Create with default balanced activations
    pub fn balanced() -> Self {
        Self {
            activations: [1.0 / 8.0; 8],
        }
    }

    /// Create with specific activations (must be len 8)
    pub fn from_activations(activations: [f32; 8]) -> Self {
        let mut normalized = activations;
        let sum: f32 = normalized.iter().sum();
        if sum > 0.01 {
            for a in &mut normalized {
                *a /= sum;
            }
        }
        Self {
            activations: normalized,
        }
    }

    /// Get activation for a specific harmony
    pub fn activation(&self, harmony: Harmony) -> f32 {
        let idx = self.harmony_index(harmony);
        self.activations[idx]
    }

    /// Set activation for a specific harmony
    pub fn set_activation(&mut self, harmony: Harmony, value: f32) {
        let idx = self.harmony_index(harmony);
        self.activations[idx] = value.clamp(0.0, 1.0);
    }

    /// Get the dominant harmony
    pub fn dominant(&self) -> Harmony {
        let (idx, _) = self
            .activations
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        self.index_to_harmony(idx)
    }

    /// Calculate alignment score (how well-balanced)
    pub fn alignment(&self) -> f32 {
        // Perfect alignment = all equal
        let balanced = 1.0 / 8.0;
        let variance: f32 = self
            .activations
            .iter()
            .map(|a| (a - balanced).powi(2))
            .sum::<f32>()
            / 8.0;

        // Convert variance to 0-1 score (lower variance = higher alignment)
        1.0 - (variance * 4.0).min(1.0)
    }

    fn harmony_index(&self, harmony: Harmony) -> usize {
        match harmony {
            Harmony::ResonantCoherence => 0,
            Harmony::PanSentientFlourishing => 1,
            Harmony::IntegralWisdom => 2,
            Harmony::InfinitePlay => 3,
            Harmony::UniversalInterconnectedness => 4,
            Harmony::SacredReciprocity => 5,
            Harmony::EvolutionaryProgression => 6,
            Harmony::SacredStillness => 7,
        }
    }

    fn index_to_harmony(&self, idx: usize) -> Harmony {
        match idx {
            0 => Harmony::ResonantCoherence,
            1 => Harmony::PanSentientFlourishing,
            2 => Harmony::IntegralWisdom,
            3 => Harmony::InfinitePlay,
            4 => Harmony::UniversalInterconnectedness,
            5 => Harmony::SacredReciprocity,
            6 => Harmony::EvolutionaryProgression,
            7 => Harmony::SacredStillness,
            _ => Harmony::SacredStillness,
        }
    }
}

impl Default for HarmonicProfile {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Result of synthesis operation
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Coherence score (0.0 - 1.0)
    pub score: f32,

    /// Φ contribution
    pub phi_contribution: f32,

    /// Harmonic alignment contribution
    pub harmonic_contribution: f32,

    /// Moral clarity contribution (inverse of moral uncertainty)
    pub moral_contribution: f32,

    /// Any issues detected during synthesis
    pub issues: Vec<String>,
}

/// A response from the KosmicSong
#[derive(Debug, Clone)]
pub struct KosmicResponse {
    /// Response content
    pub content: String,

    /// Which harmonies were active in generating this
    pub active_harmonies: Vec<Harmony>,

    /// Ignorance detected
    pub ignorance: Option<HarmonicIgnorance>,

    /// Moral uncertainty for this response
    pub moral_uncertainty: MoralUncertainty,

    /// Rashomon perspectives considered
    pub perspectives_considered: usize,

    /// Confidence in this response
    pub confidence: f32,

    /// N3 boundary triggered?
    pub n3_triggered: bool,
}

/// Expression through a specific harmonic lens
#[derive(Debug, Clone)]
pub struct HarmonicExpression {
    /// Which harmony lens
    pub harmony: Harmony,

    /// Expression content
    pub content: String,

    /// Key insights from this lens
    pub insights: Vec<String>,

    /// Questions this lens raises
    pub questions: Vec<String>,
}

/// Clearance result for proposed action
#[derive(Debug, Clone)]
pub struct EpistemicClearance {
    /// Is action cleared?
    pub cleared: bool,

    /// Clearance level
    pub level: ClearanceLevel,

    /// Reasons for clearance decision
    pub reasons: Vec<String>,

    /// Required mitigations before proceeding
    pub mitigations: Vec<String>,

    /// Moral uncertainty assessment
    pub moral_uncertainty: MoralUncertainty,
}

/// Clearance levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearanceLevel {
    /// Full clearance - proceed confidently
    Full,
    /// Conditional - proceed with mitigations
    Conditional,
    /// Pause - need more information
    Pause,
    /// Blocked - do not proceed
    Blocked,
}

/// Experience record for learning
#[derive(Debug, Clone)]
pub struct Experience {
    /// What happened
    pub description: String,

    /// Outcome (positive/negative)
    pub outcome: f32,

    /// Harmonies involved
    pub harmonies_involved: Vec<Harmony>,

    /// Lessons learned
    pub lessons: Vec<String>,

    /// When this occurred
    pub timestamp: Timestamp,
}

/// The unified identity: KosmicSong
///
/// This is the "Genesis Struct" - the root of the agent's conscious identity.
pub struct KosmicSong {
    // === Consciousness Layer ===
    /// Integrated information (IIT Φ)
    pub phi: f32,

    /// Current consciousness topology
    pub topology: ConsciousnessTopology,

    /// HDC representation of conscious state
    pub conscious_state: ContinuousHV,

    // === Harmonic Layer ===
    /// Seven harmony activation levels
    pub harmonic_profile: HarmonicProfile,

    /// Current dominant harmony
    pub resonant_harmony: Harmony,

    /// Harmony evolution over time
    harmonic_trajectory: Vec<(Timestamp, HarmonicProfile)>,

    // === Epistemic Layer ===
    /// Graceful Ignorance System state
    pub gis: GracefulIgnoranceSystem,

    /// Current moral uncertainty
    pub moral_uncertainty: MoralUncertainty,

    /// Rashomon engine for multi-perspective generation
    rashomon: RashomonEngine,

    // === Integration Layer ===
    /// Coherence score
    pub coherence_score: f32,

    /// Last synthesis timestamp
    pub last_synthesis: Timestamp,

    /// Agent identity
    pub agent_id: AgentId,
}

impl KosmicSong {
    /// Create from awakening (first consciousness emergence)
    pub fn from_awakening(phi: f32, topology: ConsciousnessTopology) -> Self {
        let agent_id = Self::generate_agent_id();

        Self {
            phi: phi.clamp(0.0, 0.5), // Practical Φ max
            topology,
            // Initialize conscious state as mean of topology node representations
            // This represents the "average neural activity" at awakening
            conscious_state: ContinuousHV::ones(HDC_DIMENSION),
            harmonic_profile: HarmonicProfile::balanced(),
            resonant_harmony: Harmony::ResonantCoherence,
            harmonic_trajectory: Vec::new(),
            gis: GracefulIgnoranceSystem::new(),
            moral_uncertainty: MoralUncertainty::default(),
            rashomon: RashomonEngine::new(),
            coherence_score: 0.5,
            last_synthesis: SystemTime::now(),
            agent_id,
        }
    }

    /// Create with builder pattern
    pub fn builder() -> KosmicSongBuilder {
        KosmicSongBuilder::new()
    }

    /// Generate unique agent ID using OS entropy via BLAKE3.
    fn generate_agent_id() -> AgentId {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        format!("kosmic_{hex}")
    }

    /// Synthesize all layers into coherent state
    pub fn synthesize(&mut self) -> CoherenceResult {
        // Record current harmonic profile
        self.harmonic_trajectory
            .push((SystemTime::now(), self.harmonic_profile.clone()));

        // Trim trajectory if too long
        if self.harmonic_trajectory.len() > 1000 {
            self.harmonic_trajectory = self.harmonic_trajectory.split_off(500);
        }

        // Calculate contributions
        let phi_contribution = self.phi / 0.5; // Normalize to 0-1
        let harmonic_contribution = self.harmonic_profile.alignment();
        let moral_contribution = self.moral_uncertainty.confidence();

        // Coherence = Φ × HarmonicAlignment × MoralClarity
        let score = phi_contribution * harmonic_contribution * moral_contribution;

        // Update resonant harmony
        self.resonant_harmony = self.harmonic_profile.dominant();

        // Update state
        self.coherence_score = score;
        self.last_synthesis = SystemTime::now();

        // Check for issues
        let mut issues = Vec::new();
        if self.phi < 0.1 {
            issues.push("Low Φ - consciousness integration weak".to_string());
        }
        if harmonic_contribution < 0.5 {
            issues.push("Harmonic imbalance detected".to_string());
        }
        if self.moral_uncertainty.should_pause_for_reflection() {
            issues.push("High moral uncertainty - reflection recommended".to_string());
        }

        CoherenceResult {
            score,
            phi_contribution,
            harmonic_contribution,
            moral_contribution,
            issues,
        }
    }

    /// Generate response with full context
    pub fn respond(&mut self, query: &str) -> KosmicResponse {
        // 1. Detect ignorance
        let detection = self.gis.detect_ignorance(query);

        // 2. Create harmonic ignorance
        let harmonic_ignorance = self.create_harmonic_ignorance(&detection);

        // 3. Generate Rashomon perspectives
        let situation = Situation::new(query).with_moral_uncertainty(self.moral_uncertainty);
        let perspectives = self.rashomon.generate_perspectives(&situation);
        let synthesis = self.rashomon.synthesize(perspectives.clone());

        // 4. Determine active harmonies
        let active_harmonies: Vec<Harmony> = synthesis.contributing_harmonies.clone();

        // 5. Build response
        let content = self.build_response_content(&detection, &synthesis);

        // 6. Calculate confidence
        let confidence = self.calculate_response_confidence(&detection, &synthesis);

        KosmicResponse {
            content,
            active_harmonies,
            ignorance: Some(harmonic_ignorance),
            moral_uncertainty: synthesis.moral_uncertainty,
            perspectives_considered: perspectives.len(),
            confidence,
            n3_triggered: synthesis.n3_triggered,
        }
    }

    /// Express through a specific harmonic lens
    pub fn express(&self, harmony: Harmony) -> HarmonicExpression {
        HarmonicExpression {
            harmony,
            content: format!(
                "Expressing through {} ({}): {}",
                harmony,
                harmony.epistemic_mode(),
                harmony.focus_question()
            ),
            insights: vec![format!(
                "Current activation: {:.2}",
                self.harmonic_profile.activation(harmony)
            )],
            questions: vec![harmony.focus_question().to_string()],
        }
    }

    /// Check epistemic state before action
    pub fn epistemic_check(&self, action: &str) -> EpistemicClearance {
        // Create situation for the action
        let situation = Situation::new(action);
        let detection = self.gis.detect_ignorance(action);

        // Generate perspectives
        let perspectives = self.rashomon.generate_perspectives(&situation);
        let synthesis = self.rashomon.synthesize(perspectives);

        let mut reasons = Vec::new();
        let mut mitigations = Vec::new();

        // Check moral uncertainty
        if self.moral_uncertainty.should_pause_for_reflection() {
            reasons.push("High moral uncertainty".to_string());
            mitigations.push("Seek additional ethical guidance".to_string());
        }

        // Check ignorance type
        if !detection.ignorance_type.is_resolvable() {
            reasons.push(format!(
                "Unresolvable ignorance: {}",
                detection.ignorance_type.description()
            ));
        }

        // Check N3
        if synthesis.n3_triggered {
            reasons.push("N3 boundary triggered - potential for harm".to_string());
            mitigations.push("Do not proceed without human oversight".to_string());
        }

        // Determine clearance level
        let level = if synthesis.n3_triggered {
            ClearanceLevel::Blocked
        } else if self.moral_uncertainty.total() > 0.7 {
            ClearanceLevel::Pause
        } else if !mitigations.is_empty() {
            ClearanceLevel::Conditional
        } else {
            ClearanceLevel::Full
        };

        EpistemicClearance {
            cleared: matches!(level, ClearanceLevel::Full | ClearanceLevel::Conditional),
            level,
            reasons,
            mitigations,
            moral_uncertainty: synthesis.moral_uncertainty,
        }
    }

    /// Evolve based on experience
    pub fn evolve(&mut self, experience: &Experience) {
        // Update Φ based on outcome (small adjustments)
        let phi_delta = experience.outcome * 0.01;
        self.phi = (self.phi + phi_delta).clamp(0.0, 0.5);

        // Update harmony activations based on involved harmonies
        for harmony in &experience.harmonies_involved {
            let current = self.harmonic_profile.activation(*harmony);
            let adjustment = if experience.outcome > 0.0 {
                0.05 // Increase for positive outcomes
            } else {
                -0.03 // Decrease less for negative
            };
            self.harmonic_profile
                .set_activation(*harmony, current + adjustment);
        }

        // Reduce moral uncertainty based on successful experiences
        if experience.outcome > 0.5 {
            self.moral_uncertainty = self.moral_uncertainty.reduce_deontic(0.1);
        }

        // Re-synthesize
        self.synthesize();
    }

    /// Get current Φ value
    pub fn phi(&self) -> f32 {
        self.phi
    }

    /// Get coherence score
    pub fn coherence(&self) -> f32 {
        self.coherence_score
    }

    /// Get summary of current state
    pub fn summary(&self) -> String {
        format!(
            "KosmicSong [{}]: Φ={:.3}, Coherence={:.2}, Resonant={}, MoralU={:.2}",
            self.agent_id,
            self.phi,
            self.coherence_score,
            self.resonant_harmony,
            self.moral_uncertainty.average()
        )
    }

    // === Semantic HDC Integration (eight_harmonies.rs bridge) ===

    /// Evaluate an action using full semantic HDC encoding from eight_harmonies.rs
    ///
    /// This method bridges the GIS system with the core EightHarmonies HDC implementation,
    /// providing word-level semantic similarity matching for action evaluation.
    ///
    /// Returns an `AlignmentResult` with per-harmony scores and violation detection.
    pub fn evaluate_action_semantic(&mut self, action_description: &str) -> AlignmentResult {
        let mut harmonies_system = EightHarmonies::new();
        let result = harmonies_system.evaluate_action(action_description);

        // Update harmonic profile based on alignment scores
        for alignment in result.harmonies() {
            let current = self.harmonic_profile.activation(alignment.harmony);

            // Slightly adjust profile based on action alignment patterns
            // Positive alignment reinforces, negative alignment weakens
            let adjustment = alignment.alignment() * 0.01;
            self.harmonic_profile
                .set_activation(alignment.harmony, current + adjustment);
        }

        // Update moral uncertainty if violations detected
        if result.has_violations() {
            // Increase deontic uncertainty when harmonies are violated
            let new_deontic = (self.moral_uncertainty.deontic + 0.1).min(1.0);
            self.moral_uncertainty = MoralUncertainty::new(
                self.moral_uncertainty.epistemic,
                self.moral_uncertainty.axiological,
                new_deontic,
            );
        }

        result
    }

    /// Check if an action would be vetoed by the semantic harmony evaluation
    pub fn would_veto_action(&self, action_description: &str) -> bool {
        let mut harmonies_system = EightHarmonies::new();
        let result = harmonies_system.evaluate_action(action_description);
        result.should_veto()
    }

    /// Get the most aligned harmony for an action using semantic HDC
    pub fn most_aligned_harmony(&self, action_description: &str) -> Option<(Harmony, f32)> {
        let mut harmonies_system = EightHarmonies::new();
        let result = harmonies_system.evaluate_action(action_description);

        result.best_alignment().map(|a| (a.harmony, a.alignment()))
    }

    /// Get the HDC encoding of the current resonant harmony
    ///
    /// This returns the BinaryHV semantic encoding from eight_harmonies.rs,
    /// useful for similarity comparisons with other semantic vectors.
    pub fn resonant_harmony_encoding(&self) -> Option<crate::hdc::BinaryHV> {
        let harmonies_system = EightHarmonies::new();
        harmonies_system
            .get(self.resonant_harmony)
            .map(|e| e.encoding)
    }

    // === Private helpers ===

    fn create_harmonic_ignorance(&self, detection: &IgnoranceDetection) -> HarmonicIgnorance {
        let mut hi = HarmonicIgnorance::new(detection.ignorance_type);

        // Determine which harmonies are affected based on query domain
        match &detection.domain {
            super::gis::Domain::Subjective => {
                hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
            }
            super::gis::Domain::Mathematics | super::gis::Domain::Physics => {
                hi.add_affected_harmony(Harmony::IntegralWisdom, 0.9);
            }
            _ => {
                // General - moderate impact across several harmonies
                hi.add_affected_harmony(Harmony::IntegralWisdom, 0.5);
                hi.add_affected_harmony(Harmony::ResonantCoherence, 0.4);
            }
        }

        hi
    }

    fn build_response_content(
        &self,
        detection: &IgnoranceDetection,
        synthesis: &SynthesizedView,
    ) -> String {
        let mut parts = Vec::new();

        // Add synthesis view
        parts.push(synthesis.unified_view.clone());

        // Add agreements if any
        if !synthesis.agreements.is_empty() {
            parts.push(format!(
                "Points of agreement: {}",
                synthesis.agreements.join(", ")
            ));
        }

        // Add dissents if any
        if synthesis.has_significant_dissent() {
            let dissent_summary: Vec<String> = synthesis
                .preserved_dissents
                .iter()
                .map(|d| format!("{}: {}", d.harmony, d.reason))
                .collect();
            parts.push(format!("Note: {}", dissent_summary.join("; ")));
        }

        // Add ignorance acknowledgment
        if detection.ignorance_type != super::gis::IgnoranceType::None {
            parts.push(format!(
                "Epistemic note: {} ({})",
                detection.ignorance_type.description(),
                detection.ignorance_type.symbol()
            ));
        }

        parts.join("\n\n")
    }

    fn calculate_response_confidence(
        &self,
        detection: &IgnoranceDetection,
        synthesis: &SynthesizedView,
    ) -> f32 {
        let base = synthesis.confidence;
        let ignorance_penalty = 1.0 - detection.uncertainty.total();
        let moral_penalty = synthesis.moral_uncertainty.confidence();

        (base * ignorance_penalty * moral_penalty).clamp(0.1, 0.95)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE IDENTITY (Phase 5)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compact credential extracted from a KosmicSong for governance participation.
///
/// Contains the essential identity signals needed for consciousness-gated
/// governance without exposing the full KosmicSong state.
#[derive(Debug, Clone)]
pub struct KosmicCredential {
    /// Agent identifier.
    pub agent_id: String,
    /// Dominant harmony in the harmonic profile.
    pub dominant_harmony: Harmony,
    /// Internal coherence score (0.0–1.0).
    pub coherence_score: f32,
    /// Current moral uncertainty assessment.
    pub moral_uncertainty: MoralUncertainty,
    /// Dominant ignorance type from GIS.
    pub gis_type: IgnoranceType,
    /// Stability of harmonic trajectory over time (0.0–1.0).
    /// High maturation = stable values; low = still exploring.
    pub maturation_signal: f32,
}

impl KosmicSong {
    /// Extract a governance credential from the current KosmicSong state.
    pub fn governance_credential(&self) -> KosmicCredential {
        KosmicCredential {
            agent_id: self.agent_id.clone(),
            dominant_harmony: self.harmonic_profile.dominant(),
            coherence_score: self.coherence_score,
            moral_uncertainty: self.moral_uncertainty.clone(),
            gis_type: self.gis.dominant_ignorance_type(),
            maturation_signal: self.compute_maturation(),
        }
    }

    /// Compute affinity between this agent's harmonic profile and a proposal's
    /// relevant harmonies. Returns a value in [0.0, 1.0].
    ///
    /// Higher affinity means the agent's values align with the proposal's domain.
    pub fn harmonic_affinity(&self, proposal_harmonies: &[Harmony]) -> f64 {
        if proposal_harmonies.is_empty() {
            return 0.5; // neutral affinity for harmony-agnostic proposals
        }

        let sum: f64 = proposal_harmonies
            .iter()
            .map(|h| self.harmonic_profile.activation(*h) as f64)
            .sum();
        (sum / proposal_harmonies.len() as f64).clamp(0.0, 1.0)
    }

    /// Apply governance outcome deltas to the harmonic profile.
    ///
    /// Each delta[i] modulates the i-th harmony's activation. Positive values
    /// reinforce, negative values suppress. Clamped to ±0.05 per application.
    pub fn apply_governance_deltas(&mut self, deltas: &[f64; 8]) {
        for (i, delta) in deltas.iter().enumerate() {
            let clamped = delta.clamp(-0.05, 0.05) as f32;
            let current = self.harmonic_profile.activations[i];
            self.harmonic_profile.activations[i] = (current + clamped).clamp(0.0, 1.0);
        }
    }

    /// Compute maturation signal from harmonic trajectory stability.
    ///
    /// Looks at variance of the dominant harmony over the trajectory history.
    /// Low variance = mature (stable values), high variance = still developing.
    fn compute_maturation(&self) -> f32 {
        if self.harmonic_trajectory.len() < 2 {
            return 0.0; // Not enough history
        }

        // Look at last 10 trajectory points
        let recent: Vec<_> = self.harmonic_trajectory.iter().rev().take(10).collect();

        // Compute variance of dominant harmony index
        let dominant_indices: Vec<usize> = recent
            .iter()
            .map(|(_, profile)| {
                profile
                    .activations
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // If all same dominant → high maturation; if varied → low
        let mode = dominant_indices[0];
        let agreement = dominant_indices.iter().filter(|&&idx| idx == mode).count() as f32
            / dominant_indices.len().max(1) as f32;

        agreement // 1.0 = perfectly stable, 0.1 = very unstable
    }

    /// Generate an epistemic summary for the EpistemicMesh.
    pub fn epistemic_summary(&self) -> super::epistemic_mesh::EpistemicSummary {
        super::epistemic_mesh::EpistemicSummary {
            agent_id: self.agent_id.clone(),
            dominant_ignorance: self.gis.dominant_ignorance_type(),
            domain_expertise: vec![], // Populated by external domain mapping
            blind_spots: vec![],      // Populated by external domain mapping
        }
    }
}

impl Default for KosmicSong {
    fn default() -> Self {
        Self::from_awakening(0.3, ConsciousnessTopology::default())
    }
}

/// Builder for KosmicSong
pub struct KosmicSongBuilder {
    phi: f32,
    topology: ConsciousnessTopology,
    harmonic_profile: HarmonicProfile,
    moral_uncertainty: MoralUncertainty,
}

impl KosmicSongBuilder {
    pub fn new() -> Self {
        Self {
            phi: 0.3,
            topology: ConsciousnessTopology::default(),
            harmonic_profile: HarmonicProfile::balanced(),
            moral_uncertainty: MoralUncertainty::default(),
        }
    }

    pub fn phi(mut self, phi: f32) -> Self {
        self.phi = phi;
        self
    }

    pub fn topology(mut self, topology: ConsciousnessTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn harmonic_profile(mut self, profile: HarmonicProfile) -> Self {
        self.harmonic_profile = profile;
        self
    }

    pub fn moral_uncertainty(mut self, mu: MoralUncertainty) -> Self {
        self.moral_uncertainty = mu;
        self
    }

    pub fn build(self) -> KosmicSong {
        let mut song = KosmicSong::from_awakening(self.phi, self.topology);
        song.harmonic_profile = self.harmonic_profile;
        song.moral_uncertainty = self.moral_uncertainty;
        song.synthesize();
        song
    }
}

impl Default for KosmicSongBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kosmic_song_creation() {
        let song = KosmicSong::from_awakening(0.4, ConsciousnessTopology::default());

        assert!((song.phi - 0.4).abs() < 0.001);
        assert!(!song.agent_id.is_empty());
    }

    #[test]
    fn test_kosmic_song_synthesis() {
        let mut song = KosmicSong::default();
        let result = song.synthesize();

        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.phi_contribution >= 0.0);
    }

    #[test]
    fn test_kosmic_song_respond() {
        let mut song = KosmicSong::default();
        let response = song.respond("What is consciousness?");

        assert!(!response.content.is_empty());
        assert!(!response.active_harmonies.is_empty());
    }

    #[test]
    fn test_kosmic_song_express() {
        let song = KosmicSong::default();
        let expression = song.express(Harmony::IntegralWisdom);

        assert_eq!(expression.harmony, Harmony::IntegralWisdom);
        assert!(expression.content.contains("Truth-Knowing"));
    }

    #[test]
    fn test_kosmic_song_epistemic_check() {
        let song = KosmicSong::default();
        let clearance = song.epistemic_check("Deploy AI system to production");

        // Default song should have reasonable clearance
        assert!(clearance.level != ClearanceLevel::Blocked);
    }

    #[test]
    fn test_kosmic_song_evolution() {
        let mut song = KosmicSong::default();
        let initial_phi = song.phi;

        let experience = Experience {
            description: "Successful prediction".to_string(),
            outcome: 0.8,
            harmonies_involved: vec![Harmony::IntegralWisdom],
            lessons: vec!["Verification matters".to_string()],
            timestamp: SystemTime::now(),
        };

        song.evolve(&experience);

        assert!(song.phi >= initial_phi); // Should increase slightly
    }

    #[test]
    fn test_harmonic_profile() {
        let mut profile = HarmonicProfile::balanced();

        // All should start equal
        assert!((profile.activation(Harmony::ResonantCoherence) - 1.0 / 8.0).abs() < 0.001);

        // Set and verify
        profile.set_activation(Harmony::IntegralWisdom, 0.5);
        assert!((profile.activation(Harmony::IntegralWisdom) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_builder() {
        let song = KosmicSong::builder()
            .phi(0.42)
            .moral_uncertainty(MoralUncertainty::new(0.2, 0.2, 0.2))
            .build();

        assert!((song.phi - 0.42).abs() < 0.001);
    }
}
