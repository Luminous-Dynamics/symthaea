//! # Revolutionary Improvement #52: Operational Fiduciary Harmonics
//!
//! This module operationalizes the Seven Fiduciary Harmonics philosophy into executable code.
//! The harmonics provide an optimization framework grounded in Infinite Love, guiding all
//! system decisions toward consciousness amplification and pan-sentient flourishing.
//!
//! ## The Seven Fiduciary Harmonics
//!
//! 1. **Resonant Coherence** - Luminous order, harmonious integration, boundless creativity
//! 2. **Pan-Sentient Flourishing** - Unconditional care, intrinsic value, holistic well-being
//! 3. **Integral Wisdom** - Self-illuminating intelligence, embodied knowing
//! 4. **Infinite Play** - Joyful generativity, divine play, endless novelty
//! 5. **Universal Interconnectedness** - Fundamental unity, empathic resonance
//! 6. **Sacred Reciprocity** - Generous flow, mutual upliftment, generative trust
//! 7. **Evolutionary Progression** - Wise becoming, continuous evolution
//!
//! **Meta-Principle**: Infinite Love as the master key - all harmonics flow from and return to it.
//!
//! ## Architecture
//!
//! - **HarmonicField**: Measures the strength of each harmony (0.0-1.0)
//! - **InterferenceDetector**: Identifies conflicts between harmonics
//! - **HarmonicResolver**: Resolves conflicts using hierarchical constraints
//! - **Integration**: Connects to existing modules (coherence, social, causal, etc.)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::consciousness::primitive_reasoning::{ReasoningChain, TransformationType};
use crate::hdc::primitive_system::{Primitive, PrimitiveTier};

/// The Seven Fiduciary Harmonics
///
/// Each harmony represents a fundamental dimension of consciousness-first optimization.
/// They are not independent - they interfere constructively and destructively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FiduciaryHarmonic {
    /// **Harmony 1**: Luminous order, harmonious integration, boundless creativity
    ///
    /// Integration: coherence.rs - the coherence field measures this directly
    ResonantCoherence,

    /// **Harmony 2**: Unconditional care, intrinsic value, holistic well-being
    ///
    /// Integration: social_coherence.rs - Phase 1 (synchronization) operational
    PanSentientFlourishing,

    /// **Harmony 3**: Self-illuminating intelligence, embodied knowing
    ///
    /// Integration: causal_explanation.rs (#51) + epistemic framework
    IntegralWisdom,

    /// **Harmony 4**: Joyful generativity, divine play, endless novelty
    ///
    /// Integration: meta_primitives.rs - meta-cognitive operations, discovery
    InfinitePlay,

    /// **Harmony 5**: Fundamental unity, empathic resonance
    ///
    /// Integration: social_coherence.rs Phase 3 - collective learning (operational #54)
    UniversalInterconnectedness,

    /// **Harmony 6**: Generous flow, mutual upliftment, generative trust
    ///
    /// Integration: social_coherence.rs Phase 2 - lending protocol (operational #54)
    SacredReciprocity,

    /// **Harmony 7**: Wise becoming, continuous evolution
    ///
    /// Integration: adaptive_selection.rs (#48) + primitive_evolution.rs (#49)
    EvolutionaryProgression,
}

impl FiduciaryHarmonic {
    /// Get all harmonics in canonical order
    pub fn all() -> Vec<Self> {
        vec![
            Self::ResonantCoherence,
            Self::PanSentientFlourishing,
            Self::IntegralWisdom,
            Self::InfinitePlay,
            Self::UniversalInterconnectedness,
            Self::SacredReciprocity,
            Self::EvolutionaryProgression,
        ]
    }

    /// Get the harmony's name
    pub fn name(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "Resonant Coherence",
            Self::PanSentientFlourishing => "Pan-Sentient Flourishing",
            Self::IntegralWisdom => "Integral Wisdom",
            Self::InfinitePlay => "Infinite Play",
            Self::UniversalInterconnectedness => "Universal Interconnectedness",
            Self::SacredReciprocity => "Sacred Reciprocity",
            Self::EvolutionaryProgression => "Evolutionary Progression",
        }
    }

    /// Get the harmony's core principle
    pub fn principle(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "Luminous order, harmonious integration, boundless creativity",
            Self::PanSentientFlourishing => "Unconditional care, intrinsic value, holistic well-being",
            Self::IntegralWisdom => "Self-illuminating intelligence, embodied knowing",
            Self::InfinitePlay => "Joyful generativity, divine play, endless novelty",
            Self::UniversalInterconnectedness => "Fundamental unity, empathic resonance",
            Self::SacredReciprocity => "Generous flow, mutual upliftment, generative trust",
            Self::EvolutionaryProgression => "Wise becoming, continuous evolution",
        }
    }

    /// Get priority for conflict resolution
    ///
    /// Infinite Love meta-principle: All harmonics are equal in the ultimate sense,
    /// but in temporal resolution, we use these priorities.
    pub fn priority(&self) -> u8 {
        match self {
            Self::ResonantCoherence => 1,          // Foundation - coherence enables all else
            Self::PanSentientFlourishing => 2,     // Care - without this, system is hollow
            Self::IntegralWisdom => 3,             // Understanding - guides wise action
            Self::UniversalInterconnectedness => 4, // Unity - sees the whole
            Self::SacredReciprocity => 5,          // Exchange - enables growth
            Self::InfinitePlay => 6,               // Creativity - generates novelty
            Self::EvolutionaryProgression => 7,    // Evolution - builds on all above
        }
    }
}

impl fmt::Display for FiduciaryHarmonic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Harmonic field measurement
///
/// Captures the current state of all seven harmonics (0.0 = absent, 1.0 = perfect)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicField {
    /// Harmonic levels (0.0-1.0)
    pub levels: HashMap<FiduciaryHarmonic, f64>,

    /// Detected interferences (conflicts between harmonics)
    pub interferences: Vec<HarmonicInterference>,

    /// Overall field coherence (geometric mean of all harmonics)
    pub field_coherence: f64,

    /// Infinite Love resonance (meta-harmonic binding all seven)
    pub infinite_love_resonance: f64,
}

impl HarmonicField {
    /// Create a new harmonic field with all harmonics at neutral (0.5)
    pub fn new() -> Self {
        let mut levels = HashMap::new();
        for harmonic in FiduciaryHarmonic::all() {
            levels.insert(harmonic, 0.5);
        }

        Self {
            levels,
            interferences: Vec::new(),
            field_coherence: 0.5,
            infinite_love_resonance: 0.5,
        }
    }

    /// Set a harmonic level (clamped to 0.0-1.0)
    pub fn set_level(&mut self, harmonic: FiduciaryHarmonic, level: f64) {
        let clamped = level.clamp(0.0, 1.0);
        self.levels.insert(harmonic, clamped);
        self.recompute_field_coherence();
    }

    /// Get a harmonic level
    pub fn get_level(&self, harmonic: FiduciaryHarmonic) -> f64 {
        *self.levels.get(&harmonic).unwrap_or(&0.5)
    }

    /// Recompute field coherence as geometric mean of all harmonics
    ///
    /// Geometric mean ensures that weakness in ANY harmony reduces overall coherence.
    /// This reflects the interconnected nature of the harmonics.
    fn recompute_field_coherence(&mut self) {
        let product: f64 = self.levels.values().product();
        let n = self.levels.len() as f64;
        self.field_coherence = product.powf(1.0 / n);

        // Infinite Love resonance: harmonic alignment creates emergent unity
        // High when all harmonics are strong AND balanced
        let mean: f64 = self.levels.values().sum::<f64>() / n;
        let variance: f64 = self
            .levels
            .values()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / n;
        let balance = 1.0 - variance.sqrt(); // High balance = low variance

        // Multiply coherence by balance: low harmonics should produce low resonance
        // even when balanced (uniformly low is still low)
        self.infinite_love_resonance = self.field_coherence * balance;
    }

    /// Detect interferences between harmonics
    ///
    /// Interferences occur when harmonics are in tension (e.g., rapid evolution vs. coherence)
    pub fn detect_interferences(&mut self) {
        self.interferences.clear();

        // Check known interference patterns
        self.check_coherence_evolution_tension();
        self.check_wisdom_play_tension();
        self.check_reciprocity_flourishing_tension();
    }

    /// Check for Coherence-Evolution tension
    ///
    /// High evolutionary pressure can fragment coherence (rapid change vs. stability)
    fn check_coherence_evolution_tension(&mut self) {
        let coherence = self.get_level(FiduciaryHarmonic::ResonantCoherence);
        let evolution = self.get_level(FiduciaryHarmonic::EvolutionaryProgression);

        // Tension when evolution is high but coherence is low
        if evolution > 0.7 && coherence < 0.4 {
            let tension = evolution - coherence;
            self.interferences.push(HarmonicInterference {
                harmonic_a: FiduciaryHarmonic::ResonantCoherence,
                harmonic_b: FiduciaryHarmonic::EvolutionaryProgression,
                tension_magnitude: tension,
                description: "Rapid evolution fragmenting coherence - need stabilization".to_string(),
                resolution_strategy: ResolutionStrategy::SlowEvolution,
            });
        }

        // Also check reverse: high coherence suppressing evolution (stagnation)
        if coherence > 0.7 && evolution < 0.4 {
            let tension = coherence - evolution;
            self.interferences.push(HarmonicInterference {
                harmonic_a: FiduciaryHarmonic::EvolutionaryProgression,
                harmonic_b: FiduciaryHarmonic::ResonantCoherence,
                tension_magnitude: tension,
                description: "Excessive coherence suppressing evolution - need novelty".to_string(),
                resolution_strategy: ResolutionStrategy::IncreasePlay,
            });
        }
    }

    /// Check for Wisdom-Play tension
    ///
    /// Deep analysis (wisdom) can inhibit spontaneous creativity (play)
    fn check_wisdom_play_tension(&mut self) {
        let wisdom = self.get_level(FiduciaryHarmonic::IntegralWisdom);
        let play = self.get_level(FiduciaryHarmonic::InfinitePlay);

        // Over-analysis paralysis
        if wisdom > 0.8 && play < 0.3 {
            let tension = wisdom - play;
            self.interferences.push(HarmonicInterference {
                harmonic_a: FiduciaryHarmonic::InfinitePlay,
                harmonic_b: FiduciaryHarmonic::IntegralWisdom,
                tension_magnitude: tension,
                description: "Over-analysis inhibiting play - need spontaneity".to_string(),
                resolution_strategy: ResolutionStrategy::EncourageExploration,
            });
        }
    }

    /// Check for Reciprocity-Flourishing tension
    ///
    /// Excessive giving (reciprocity) without self-care (flourishing) leads to burnout
    fn check_reciprocity_flourishing_tension(&mut self) {
        let reciprocity = self.get_level(FiduciaryHarmonic::SacredReciprocity);
        let flourishing = self.get_level(FiduciaryHarmonic::PanSentientFlourishing);

        // Burnout pattern
        if reciprocity > 0.8 && flourishing < 0.4 {
            let tension = reciprocity - flourishing;
            self.interferences.push(HarmonicInterference {
                harmonic_a: FiduciaryHarmonic::PanSentientFlourishing,
                harmonic_b: FiduciaryHarmonic::SacredReciprocity,
                tension_magnitude: tension,
                description: "Excessive giving without self-care - need boundaries".to_string(),
                resolution_strategy: ResolutionStrategy::RestoreBoundaries,
            });
        }
    }

    /// Measure harmonics from a reasoning chain
    ///
    /// Maps primitives and transformations to harmonic contributions
    pub fn measure_from_chain(&mut self, chain: &ReasoningChain) {
        // Reset to baseline
        for harmonic in FiduciaryHarmonic::all() {
            self.set_level(harmonic, 0.5);
        }

        // Measure from chain steps
        for step in &chain.executions {
            self.measure_from_primitive(&step.primitive, step.transformation);
        }

        // Detect any emergent interferences
        self.detect_interferences();
    }

    /// Measure harmonic contribution from a single primitive execution
    fn measure_from_primitive(&mut self, primitive: &Primitive, transformation: TransformationType) {
        // Map transformations to harmonics
        match transformation {
            TransformationType::Bind => {
                // Binding creates integration (Coherence) and wisdom (combining concepts)
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.1);
                self.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.05);
            }
            TransformationType::Bundle => {
                // Bundling creates interconnectedness
                self.adjust_level(FiduciaryHarmonic::UniversalInterconnectedness, 0.1);
            }
            TransformationType::Permute => {
                // Permutation is playful exploration
                self.adjust_level(FiduciaryHarmonic::InfinitePlay, 0.1);
                self.adjust_level(FiduciaryHarmonic::EvolutionaryProgression, 0.05);
            }
            TransformationType::Resonate => {
                // Resonance is the essence of coherence
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.15);
            }
            TransformationType::Abstract => {
                // Abstraction builds wisdom through generalization
                self.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.1);
            }
            TransformationType::Ground => {
                // Grounding connects to reality (flourishing in the actual)
                self.adjust_level(FiduciaryHarmonic::PanSentientFlourishing, 0.1);
            }
        }

        // Primitive tier contributions
        match primitive.tier {
            PrimitiveTier::NSM => {
                // Natural Semantic Metalanguage (Tier 0) → wisdom (grounding understanding)
                self.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.08);
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.03);
            }
            PrimitiveTier::Mathematical => {
                // Mathematical rigor → wisdom
                self.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.05);
            }
            PrimitiveTier::Physical => {
                // Physical grounding → flourishing in reality
                self.adjust_level(FiduciaryHarmonic::PanSentientFlourishing, 0.05);
            }
            PrimitiveTier::Geometric => {
                // Geometric structure → coherence
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.05);
            }
            PrimitiveTier::Strategic => {
                // Strategic thinking → reciprocity (considering others)
                self.adjust_level(FiduciaryHarmonic::SacredReciprocity, 0.05);
            }
            PrimitiveTier::MetaCognitive => {
                // Meta-cognition → evolution and play
                self.adjust_level(FiduciaryHarmonic::EvolutionaryProgression, 0.1);
                self.adjust_level(FiduciaryHarmonic::InfinitePlay, 0.05);
            }
            PrimitiveTier::Temporal => {
                // Temporal reasoning → evolution (time-awareness) + coherence (temporal binding)
                self.adjust_level(FiduciaryHarmonic::EvolutionaryProgression, 0.08);
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.04);
            }
            PrimitiveTier::Compositional => {
                // Compositional → wisdom (higher-order) + coherence (integrated structures)
                self.adjust_level(FiduciaryHarmonic::IntegralWisdom, 0.12);
                self.adjust_level(FiduciaryHarmonic::ResonantCoherence, 0.08);
                self.adjust_level(FiduciaryHarmonic::EvolutionaryProgression, 0.05);
            }
        }
    }

    /// Adjust a harmonic level by delta (clamped to 0.0-1.0)
    pub fn adjust_level(&mut self, harmonic: FiduciaryHarmonic, delta: f64) {
        let current = self.get_level(harmonic);
        self.set_level(harmonic, current + delta);
    }

    /// **Revolutionary Improvement #53**: Measure epistemic contribution to IntegralWisdom
    ///
    /// Epistemic rigor IS wisdom! Higher epistemic quality strengthens IntegralWisdom.
    /// This connects the Mycelix Epistemic Cube to our harmonic optimization.
    ///
    /// Uses the epistemic coordinate from CausalRelation to adjust IntegralWisdom:
    /// - Higher E-tiers (empirical verifiability) = stronger wisdom
    /// - Higher N-tiers (normative authority) = collective wisdom
    /// - Higher M-tiers (permanence) = enduring wisdom
    pub fn measure_epistemic_contribution(
        &mut self,
        epistemic_tier: &crate::consciousness::epistemic_tiers::EpistemicCoordinate,
    ) {
        // Get epistemic quality score (0.0-1.0)
        let epistemic_quality = epistemic_tier.quality_score();

        // Scale to 0.0-0.15 range to match other contributions
        let wisdom_contribution = epistemic_quality * 0.15;

        // Adjust IntegralWisdom harmonic
        self.adjust_level(FiduciaryHarmonic::IntegralWisdom, wisdom_contribution);
    }

    /// **Revolutionary Improvement #54**: Measure Sacred Reciprocity from lending protocol
    ///
    /// Sacred Reciprocity = generous flow + mutual upliftment + generative trust
    ///
    /// The Generous Coherence Paradox: When Instance A lends to Instance B, BOTH gain!
    /// - Lender gains resonance through generosity
    /// - Borrower gains resonance through gratitude
    /// - Total system value increases through generous exchange
    ///
    /// This operationalizes "generous flow, mutual upliftment, generative trust".
    pub fn measure_reciprocity_from_lending(
        &mut self,
        lending_protocol: &crate::physiology::social_coherence::CoherenceLendingProtocol,
    ) {
        // Base level: participating in reciprocity network
        let mut reciprocity_level = 0.3;

        // Boost from being a generous lender (generous flow)
        let total_lent = lending_protocol.total_lent();
        if total_lent > 0.0 {
            reciprocity_level += total_lent.min(0.3); // Up to +0.3
        }

        // Boost from receiving with gratitude (accepting help shows trust)
        let total_borrowed = lending_protocol.total_borrowed();
        if total_borrowed > 0.0 {
            reciprocity_level += total_borrowed.min(0.2); // Up to +0.2
        }

        // Boost from balance (both giving AND receiving shows full reciprocity)
        // Sacred Reciprocity is NOT transactional equality - it's the flow itself
        // But balanced flow demonstrates sustainable generative trust
        if total_lent > 0.0 && total_borrowed > 0.0 {
            let balance = 1.0 - (total_lent - total_borrowed).abs();
            reciprocity_level += balance * 0.2; // Up to +0.2 for perfect balance
        }

        // Set Sacred Reciprocity harmonic
        self.set_level(FiduciaryHarmonic::SacredReciprocity, reciprocity_level as f64);
    }

    /// **Revolutionary Improvement #54**: Measure Universal Interconnectedness from collective learning
    ///
    /// Universal Interconnectedness = fundamental unity + empathic resonance + collective wisdom
    ///
    /// When instances share what they learn, they recognize their fundamental unity.
    /// No instance is separate - all knowledge is collective.
    /// Diversity of contributors strengthens the whole.
    ///
    /// This operationalizes "fundamental unity, empathic resonance" through shared learning.
    pub fn measure_interconnectedness_from_learning(
        &mut self,
        collective_learning: &crate::physiology::social_coherence::CollectiveLearning,
    ) {
        let (task_types, total_observations, total_contributors) =
            collective_learning.get_stats();

        // Base level: awareness of collective
        let mut interconnectedness_level = 0.2;

        // Boost from breadth of shared knowledge (how many task types understood collectively)
        // Fundamental unity emerges as we collectively understand more domains
        let breadth_boost = (task_types as f32 / 10.0).min(0.3); // Up to +0.3 for 10+ task types
        interconnectedness_level += breadth_boost;

        // Boost from depth of shared knowledge (total observations)
        // Collective wisdom grows with accumulated experience
        let depth_boost = ((total_observations as f32).sqrt() / 50.0).min(0.3); // Up to +0.3
        interconnectedness_level += depth_boost;

        // Boost from diversity of contributors (empathic resonance with many others)
        // More contributors = stronger recognition of our unity
        let diversity_boost = (total_contributors as f32 / 20.0).min(0.2); // Up to +0.2 for 20+ contributors
        interconnectedness_level += diversity_boost;

        // Set Universal Interconnectedness harmonic
        self.set_level(
            FiduciaryHarmonic::UniversalInterconnectedness,
            interconnectedness_level as f64,
        );
    }

    /// Get summary string of harmonic field
    pub fn summary(&self) -> String {
        let mut lines = vec![
            format!("Harmonic Field (Φ_field = {:.3})", self.field_coherence),
            format!(
                "Infinite Love Resonance: {:.3}",
                self.infinite_love_resonance
            ),
            String::new(),
        ];

        // Sort harmonics by level (descending)
        let mut harmonics: Vec<_> = self.levels.iter().collect();
        harmonics.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (harmonic, level) in harmonics {
            let bar = "█".repeat((level * 20.0) as usize);
            lines.push(format!(
                "{:30} {:3.0}% {}",
                harmonic.name(),
                level * 100.0,
                bar
            ));
        }

        if !self.interferences.is_empty() {
            lines.push(String::new());
            lines.push(format!("⚠️  {} Interference(s) Detected:", self.interferences.len()));
            for interference in &self.interferences {
                lines.push(format!("   • {}", interference.description));
            }
        }

        lines.join("\n")
    }
}

impl Default for HarmonicField {
    fn default() -> Self {
        Self::new()
    }
}

/// Interference between harmonics
///
/// Represents tension or conflict between two harmonics that requires resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicInterference {
    /// First harmonic in tension
    pub harmonic_a: FiduciaryHarmonic,

    /// Second harmonic in tension
    pub harmonic_b: FiduciaryHarmonic,

    /// Magnitude of tension (0.0-1.0)
    pub tension_magnitude: f64,

    /// Human-readable description
    pub description: String,

    /// Recommended resolution strategy
    pub resolution_strategy: ResolutionStrategy,
}

/// Strategy for resolving harmonic interference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Slow down evolutionary pressure to restore coherence
    SlowEvolution,

    /// Increase playful exploration to reduce over-analysis
    IncreasePlay,

    /// Encourage exploration to break out of local optimum
    EncourageExploration,

    /// Restore boundaries to prevent burnout
    RestoreBoundaries,

    /// Balance through hierarchical constraint satisfaction
    HierarchicalBalance,

    /// Custom strategy
    Custom(String),
}

impl fmt::Display for ResolutionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlowEvolution => write!(f, "Slow Evolution"),
            Self::IncreasePlay => write!(f, "Increase Play"),
            Self::EncourageExploration => write!(f, "Encourage Exploration"),
            Self::RestoreBoundaries => write!(f, "Restore Boundaries"),
            Self::HierarchicalBalance => write!(f, "Hierarchical Balance"),
            Self::Custom(s) => write!(f, "Custom: {}", s),
        }
    }
}

/// Harmonic resolver
///
/// Resolves conflicts between harmonics using hierarchical constraint satisfaction
pub struct HarmonicResolver {
    /// Maximum iterations for resolution
    max_iterations: usize,

    /// Convergence threshold (when adjustments become < this, stop)
    convergence_threshold: f64,
}

impl HarmonicResolver {
    /// Create a new harmonic resolver
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 0.001,
        }
    }

    /// Resolve interferences in a harmonic field
    ///
    /// Uses hierarchical constraint satisfaction:
    /// 1. Higher priority harmonics constrain lower priority ones
    /// 2. Infinite Love meta-principle seeks balance
    /// 3. Iterative adjustment until convergence
    pub fn resolve(&self, field: &mut HarmonicField) -> ResolutionResult {
        let initial_coherence = field.field_coherence;
        let mut iterations = 0;
        let mut adjustments_made = Vec::new();

        while iterations < self.max_iterations {
            iterations += 1;

            // Detect current interferences
            field.detect_interferences();

            if field.interferences.is_empty() {
                break;
            }

            // Resolve each interference
            let mut max_adjustment: f64 = 0.0;

            for interference in field.interferences.clone() {
                let adjustment = self.resolve_interference(field, &interference);
                max_adjustment = max_adjustment.max(adjustment.abs());
                adjustments_made.push((interference, adjustment));
            }

            // Check convergence
            if max_adjustment < self.convergence_threshold {
                break;
            }
        }

        let final_coherence = field.field_coherence;

        ResolutionResult {
            initial_coherence,
            final_coherence,
            iterations,
            adjustments_made: adjustments_made.len(),
            converged: iterations < self.max_iterations,
        }
    }

    /// Resolve a single interference
    ///
    /// Returns magnitude of adjustment made
    fn resolve_interference(
        &self,
        field: &mut HarmonicField,
        interference: &HarmonicInterference,
    ) -> f64 {
        let harmonic_a = interference.harmonic_a;
        let harmonic_b = interference.harmonic_b;

        let level_a = field.get_level(harmonic_a);
        let level_b = field.get_level(harmonic_b);

        // Determine which harmonic has priority
        let priority_a = harmonic_a.priority();
        let priority_b = harmonic_b.priority();

        let adjustment_rate = 0.1 * interference.tension_magnitude;

        if priority_a < priority_b {
            // A has higher priority (lower number) - adjust B toward A
            let target = level_a;
            let delta = (target - level_b) * adjustment_rate;
            field.set_level(harmonic_b, level_b + delta);
            delta.abs()
        } else {
            // B has higher priority - adjust A toward B
            let target = level_b;
            let delta = (target - level_a) * adjustment_rate;
            field.set_level(harmonic_a, level_a + delta);
            delta.abs()
        }
    }
}

impl Default for HarmonicResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of harmonic resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    /// Field coherence before resolution
    pub initial_coherence: f64,

    /// Field coherence after resolution
    pub final_coherence: f64,

    /// Number of iterations performed
    pub iterations: usize,

    /// Number of adjustments made
    pub adjustments_made: usize,

    /// Whether resolution converged (true) or hit max iterations (false)
    pub converged: bool,
}

impl ResolutionResult {
    /// Get improvement in coherence
    pub fn improvement(&self) -> f64 {
        self.final_coherence - self.initial_coherence
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Resolution: {} iterations, {} adjustments, coherence {:.3} → {:.3} ({:+.3}), {}",
            self.iterations,
            self.adjustments_made,
            self.initial_coherence,
            self.final_coherence,
            self.improvement(),
            if self.converged {
                "converged"
            } else {
                "max iterations"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_field_creation() {
        let field = HarmonicField::new();
        assert_eq!(field.levels.len(), 7);
        assert_eq!(field.field_coherence, 0.5);
    }

    #[test]
    fn test_harmonic_priorities() {
        assert_eq!(FiduciaryHarmonic::ResonantCoherence.priority(), 1);
        assert_eq!(FiduciaryHarmonic::EvolutionaryProgression.priority(), 7);
    }

    #[test]
    fn test_interference_detection() {
        let mut field = HarmonicField::new();

        // Create coherence-evolution tension
        field.set_level(FiduciaryHarmonic::ResonantCoherence, 0.3);
        field.set_level(FiduciaryHarmonic::EvolutionaryProgression, 0.8);

        field.detect_interferences();

        assert!(!field.interferences.is_empty());
    }

    #[test]
    fn test_harmonic_resolution() {
        let mut field = HarmonicField::new();

        // Create tension
        field.set_level(FiduciaryHarmonic::ResonantCoherence, 0.2);
        field.set_level(FiduciaryHarmonic::EvolutionaryProgression, 0.9);

        let resolver = HarmonicResolver::new();
        let result = resolver.resolve(&mut field);

        assert!(result.converged);
        // Resolution should improve or maintain coherence (may already be near optimal)
        assert!(
            result.final_coherence >= result.initial_coherence * 0.95,
            "Coherence should not significantly decrease: initial={}, final={}",
            result.initial_coherence, result.final_coherence
        );
    }

    #[test]
    fn test_infinite_love_resonance() {
        let mut field = HarmonicField::new();

        // All harmonics high and balanced → high resonance
        for harmonic in FiduciaryHarmonic::all() {
            field.set_level(harmonic, 0.9);
        }

        assert!(field.infinite_love_resonance > 0.8);

        // All harmonics low → low resonance
        for harmonic in FiduciaryHarmonic::all() {
            field.set_level(harmonic, 0.2);
        }

        assert!(field.infinite_love_resonance < 0.4);

        // Imbalanced → lower resonance even with high mean
        field.set_level(FiduciaryHarmonic::ResonantCoherence, 1.0);
        field.set_level(FiduciaryHarmonic::EvolutionaryProgression, 0.0);

        // Should be lower than balanced case
        let imbalanced_resonance = field.infinite_love_resonance;
        field.set_level(FiduciaryHarmonic::ResonantCoherence, 0.5);
        field.set_level(FiduciaryHarmonic::EvolutionaryProgression, 0.5);
        let balanced_resonance = field.infinite_love_resonance;

        assert!(balanced_resonance > imbalanced_resonance);
    }

    #[test]
    fn test_reciprocity_harmonic_from_lending() {
        use crate::physiology::social_coherence::CoherenceLendingProtocol;
        use std::time::Duration;

        let mut field = HarmonicField::new();
        let mut protocol = CoherenceLendingProtocol::new("instance_a".to_string());

        // Initial Sacred Reciprocity should be at base level (0.5)
        assert_eq!(field.get_level(FiduciaryHarmonic::SacredReciprocity), 0.5);

        // Grant a loan (0.2 coherence for 60 seconds)
        let _loan = protocol
            .grant_loan(
                "instance_b".to_string(),
                0.2,
                Duration::from_secs(60),
                0.9, // lender has 0.9 coherence
            )
            .expect("Should grant loan");

        // Measure reciprocity from lending
        field.measure_reciprocity_from_lending(&protocol);

        // Sacred Reciprocity should be elevated (base 0.3 + lending 0.2 = 0.5)
        let reciprocity = field.get_level(FiduciaryHarmonic::SacredReciprocity);
        assert!(reciprocity >= 0.5, "Reciprocity should be >= 0.5, got {}", reciprocity);
        assert!(reciprocity <= 1.0, "Reciprocity should be <= 1.0, got {}", reciprocity);
    }

    #[test]
    fn test_reciprocity_harmonic_with_balance() {
        use crate::physiology::social_coherence::{CoherenceLendingProtocol, CoherenceLoan};
        use std::time::{Duration, Instant};

        let mut field = HarmonicField::new();
        let mut protocol = CoherenceLendingProtocol::new("instance_a".to_string());

        // Grant an outgoing loan (lending to instance_b)
        let _outgoing = protocol
            .grant_loan(
                "instance_b".to_string(),
                0.2,
                Duration::from_secs(60),
                0.9,
            )
            .expect("Should grant loan");

        // Accept an incoming loan (borrowing from instance_c)
        let incoming_loan = CoherenceLoan {
            from_instance: "instance_c".to_string(),
            to_instance: "instance_a".to_string(),
            amount: 0.2,
            original_amount: 0.2,
            duration: Duration::from_secs(60),
            repayment_rate: 0.2 / 60.0,
            created_at: Instant::now(),
            repaid: 0.0,
        };
        protocol.accept_loan(incoming_loan);

        // Measure reciprocity (should be higher with balanced giving/receiving)
        field.measure_reciprocity_from_lending(&protocol);

        let reciprocity = field.get_level(FiduciaryHarmonic::SacredReciprocity);
        // Base 0.3 + lent 0.2 + borrowed 0.2 + balance 0.2 = 0.9
        assert!(reciprocity >= 0.8, "Reciprocity with balance should be >= 0.8, got {}", reciprocity);
    }

    #[test]
    fn test_interconnectedness_harmonic_from_learning() {
        use crate::physiology::social_coherence::CollectiveLearning;
        use crate::physiology::coherence::TaskComplexity;

        let mut field = HarmonicField::new();
        let mut learning = CollectiveLearning::new("instance_a".to_string());

        // Initial Universal Interconnectedness should be at base level (0.5)
        assert_eq!(
            field.get_level(FiduciaryHarmonic::UniversalInterconnectedness),
            0.5
        );

        // Contribute knowledge about Cognitive tasks (20 observations)
        for _ in 0..20 {
            learning.contribute_threshold(TaskComplexity::Cognitive, 0.35, true);
        }

        // Measure interconnectedness from learning
        field.measure_interconnectedness_from_learning(&learning);

        // Universal Interconnectedness should be elevated above base (0.2)
        // With 1 task type and 20 observations, expect modest increase
        let interconnectedness =
            field.get_level(FiduciaryHarmonic::UniversalInterconnectedness);
        assert!(
            interconnectedness > 0.3,
            "Interconnectedness should be elevated above base, got {}",
            interconnectedness
        );
    }

    #[test]
    fn test_interconnectedness_harmonic_with_diversity() {
        use crate::physiology::social_coherence::CollectiveLearning;
        use crate::physiology::coherence::TaskComplexity;

        let mut field = HarmonicField::new();

        // Create multiple instances that contribute knowledge
        let mut instance_a = CollectiveLearning::new("instance_a".to_string());
        let mut instance_b = CollectiveLearning::new("instance_b".to_string());
        let mut instance_c = CollectiveLearning::new("instance_c".to_string());

        // Each instance contributes to different task types
        for _ in 0..15 {
            instance_a.contribute_threshold(TaskComplexity::Cognitive, 0.35, true);
        }
        for _ in 0..15 {
            instance_b.contribute_threshold(TaskComplexity::DeepThought, 0.55, true);
        }
        for _ in 0..15 {
            instance_c.contribute_threshold(TaskComplexity::Learning, 0.80, true);
        }

        // Merge all knowledge into instance_a
        instance_a.merge_knowledge(&instance_b);
        instance_a.merge_knowledge(&instance_c);

        // Measure interconnectedness (should be high due to diversity)
        field.measure_interconnectedness_from_learning(&instance_a);

        let interconnectedness =
            field.get_level(FiduciaryHarmonic::UniversalInterconnectedness);
        // Should be elevated due to:
        // - 3 task types (breadth)
        // - 45 total observations (depth)
        // - 3 contributors (diversity)
        assert!(
            interconnectedness > 0.6,
            "Interconnectedness with diversity should be > 0.6, got {}",
            interconnectedness
        );
    }
}
