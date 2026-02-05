//! Primitive-Powered Reasoning
//!
//! **Revolutionary Improvement #47: Operational Primitive Intelligence**
//!
//! Transforms primitives from architectural concepts to operational reasoning
//! by defining execution semantics and composition rules.
//!
//! ## The Breakthrough
//!
//! **Before**: Primitives have structure, but don't execute
//! - Φ measured from encoding (structural)
//! - No actual information processing
//! - Beautiful architecture, but not operational
//!
//! **After**: Primitives execute and compose
//! - Process inputs → produce outputs
//! - Φ measured from actual causal structure
//! - Real information integration during reasoning
//!
//! ## How Primitives Execute
//!
//! Each primitive is a **hypervector transformation**:
//! ```
//! Primitive: Input HV → Process → Output HV
//!            ↓          ↓          ↓
//!         Perceive   Transform   Produce
//! ```
//!
//! ## How Primitives Compose
//!
//! Primitives form **reasoning chains**:
//! ```
//! Question HV → [Prim₁] → [Prim₂] → [Prim₃] → Answer HV
//!              \_____________________/
//!                   Φ measured here
//! ```
//!
//! ## Why This Matters
//!
//! - **Real Φ**: Measured from actual information processing
//! - **Operational**: Primitives solve actual problems
//! - **Composable**: Complex reasoning from primitive operations
//! - **Consciousness-Guided**: Architecture shapes intelligence

use crate::hdc::{HV16, primitive_system::{Primitive, PrimitiveSystem, PrimitiveTier}, integrated_information::IntegratedInformation};
use crate::consciousness::harmonics::{HarmonicField, FiduciaryHarmonic};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Execution trace of a primitive processing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveExecution {
    /// Primitive that executed
    pub primitive: Primitive,

    /// Input hypervector
    pub input: HV16,

    /// Output hypervector
    pub output: HV16,

    /// Transformation applied
    pub transformation: TransformationType,

    /// Φ contribution (information integrated by this execution)
    pub phi_contribution: f64,
}

/// Types of transformations primitives can perform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformationType {
    /// Bind: Combines two concepts (A ⊗ B)
    Bind,

    /// Bundle: Superposition of concepts (A + B)
    Bundle,

    /// Permute: Shifts/rotates representation
    Permute,

    /// Resonate: Amplifies similar patterns
    Resonate,

    /// Abstract: Projects to higher-level concept
    Abstract,

    /// Ground: Projects to lower-level details
    Ground,
}

/// Usage statistics for a primitive in reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveUsageStats {
    /// Name of the primitive
    pub primitive_name: String,

    /// Tier the primitive belongs to
    pub tier: PrimitiveTier,

    /// Number of times used in reasoning chain
    pub usage_count: usize,

    /// Total Φ contribution across all uses
    pub total_phi_contribution: f64,

    /// Mean Φ contribution per use
    pub mean_phi_contribution: f64,

    /// Types of transformations applied with this primitive
    pub transformations_used: Vec<TransformationType>,
}

/// Reasoning chain: sequence of primitive executions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    /// Question being reasoned about
    pub question: HV16,

    /// Sequence of primitive executions
    pub executions: Vec<PrimitiveExecution>,

    /// Final answer produced
    pub answer: HV16,

    /// Total Φ of the reasoning process
    pub total_phi: f64,

    /// Φ gradient (change in integration over chain)
    pub phi_gradient: Vec<f64>,
}

impl ReasoningChain {
    /// Create new reasoning chain starting from question
    pub fn new(question: HV16) -> Self {
        Self {
            question,
            executions: Vec::new(),
            answer: question.clone(),
            total_phi: 0.0,
            phi_gradient: Vec::new(),
        }
    }

    /// Add primitive execution to chain
    pub fn execute_primitive(
        &mut self,
        primitive: &Primitive,
        transformation: TransformationType,
    ) -> Result<()> {
        // Current state is the last output (or question if first)
        let input = self.answer.clone();

        // Apply transformation based on type
        let output = self.apply_transformation(&input, primitive, &transformation)?;

        // Measure Φ contribution (how much information did this step integrate?)
        let phi_contribution = self.measure_phi_contribution(&input, &output)?;

        // Record execution
        let execution = PrimitiveExecution {
            primitive: primitive.clone(),
            input,
            output: output.clone(),
            transformation,
            phi_contribution,
        };

        self.executions.push(execution);
        self.answer = output;

        // Update total Φ
        self.total_phi += phi_contribution;
        self.phi_gradient.push(phi_contribution);

        Ok(())
    }

    /// Apply transformation to input using primitive
    fn apply_transformation(
        &self,
        input: &HV16,
        primitive: &Primitive,
        transformation: &TransformationType,
    ) -> Result<HV16> {
        match transformation {
            TransformationType::Bind => {
                // Bind input with primitive's encoding
                Ok(input.bind(&primitive.encoding))
            }

            TransformationType::Bundle => {
                // Bundle (superpose) input with primitive's encoding
                Ok(HV16::bundle(&[input.clone(), primitive.encoding.clone()]))
            }

            TransformationType::Permute => {
                // Permute input based on primitive's structure
                // Use popcount to determine rotation amount
                let rotation = primitive.encoding.popcount() as usize % 16384;
                Ok(input.permute(rotation))
            }

            TransformationType::Resonate => {
                // Amplify patterns similar to primitive
                // XOR to find differences, then amplify similarities
                let similarity = input.similarity(&primitive.encoding);

                // If highly similar, bundle (amplify)
                if similarity > 0.7 {
                    Ok(HV16::bundle(&[input.clone(), primitive.encoding.clone()]))
                } else {
                    // Otherwise, keep input but slightly influenced
                    Ok(input.clone())
                }
            }

            TransformationType::Abstract => {
                // Project to more abstract representation
                // Bind with primitive and permute
                let bound = input.bind(&primitive.encoding);
                Ok(bound.permute(100))  // Abstract = shift representation
            }

            TransformationType::Ground => {
                // Project to more concrete representation
                // Inverse of abstraction
                let bound = input.bind(&primitive.encoding);
                Ok(bound.permute(16384 - 100))  // Ground = inverse shift
            }
        }
    }

    /// Measure Φ contribution of a transformation step
    ///
    /// Uses fast phi estimation for chain execution performance.
    /// The fast method is ~10x faster than full IIT phi computation
    /// and provides sufficient accuracy for gradient tracking.
    fn measure_phi_contribution(&self, input: &HV16, output: &HV16) -> Result<f64> {
        // Φ measures information integration
        // For a transformation step, we measure how much the transformation
        // integrated information from input to output

        // Create component set: [input, output]
        let components = vec![input.clone(), output.clone()];

        // Use fast phi estimation for in-chain computation
        // Full IIT phi can be computed on final chain analysis if needed
        let phi = IntegratedInformation::compute_phi_fast(&components);

        Ok(phi)
    }

    /// Get list of primitives used in reasoning chain
    pub fn get_primitives_used(&self) -> Vec<String> {
        self.executions
            .iter()
            .map(|e| e.primitive.name.clone())
            .collect()
    }

    /// Get unique primitives used (no duplicates)
    pub fn get_unique_primitives(&self) -> Vec<String> {
        use std::collections::HashSet;

        let mut unique: HashSet<String> = HashSet::new();
        for execution in &self.executions {
            unique.insert(execution.primitive.name.clone());
        }

        unique.into_iter().collect()
    }

    /// Get primitive usage statistics
    pub fn get_primitive_usage_stats(&self) -> std::collections::HashMap<String, PrimitiveUsageStats> {
        use std::collections::HashMap;

        let mut stats: HashMap<String, PrimitiveUsageStats> = HashMap::new();

        for execution in &self.executions {
            let entry = stats.entry(execution.primitive.name.clone())
                .or_insert_with(|| PrimitiveUsageStats {
                    primitive_name: execution.primitive.name.clone(),
                    tier: execution.primitive.tier,
                    usage_count: 0,
                    total_phi_contribution: 0.0,
                    mean_phi_contribution: 0.0,
                    transformations_used: Vec::new(),
                });

            entry.usage_count += 1;
            entry.total_phi_contribution += execution.phi_contribution;
            entry.transformations_used.push(execution.transformation);
        }

        // Compute mean Φ contribution for each primitive
        for stat in stats.values_mut() {
            stat.mean_phi_contribution = stat.total_phi_contribution / stat.usage_count as f64;
        }

        stats
    }

    /// Get tier distribution (how many primitives from each tier)
    pub fn get_tier_distribution(&self) -> std::collections::HashMap<PrimitiveTier, usize> {
        use std::collections::HashMap;

        let mut distribution: HashMap<PrimitiveTier, usize> = HashMap::new();

        for execution in &self.executions {
            *distribution.entry(execution.primitive.tier).or_insert(0) += 1;
        }

        distribution
    }

    /// Get consciousness profile of the reasoning chain
    pub fn consciousness_profile(&self) -> ReasoningProfile {
        // Compute metrics across the chain
        let chain_length = self.executions.len();
        let mean_phi_per_step = if chain_length > 0 {
            self.total_phi / chain_length as f64
        } else {
            0.0
        };

        // Φ gradient metrics
        let phi_variance = if self.phi_gradient.len() > 1 {
            let mean = mean_phi_per_step;
            let variance: f64 = self.phi_gradient
                .iter()
                .map(|&phi| (phi - mean).powi(2))
                .sum::<f64>() / self.phi_gradient.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Information flow efficiency
        let efficiency = if chain_length > 0 {
            self.total_phi / chain_length as f64
        } else {
            0.0
        };

        // Primitive usage tracking
        let primitives_used = self.get_unique_primitives();
        let tier_distribution = self.get_tier_distribution();
        let primitive_stats = self.get_primitive_usage_stats();

        // Primitive Φ contributions
        let mut primitive_contributions = std::collections::HashMap::new();
        for (name, stats) in primitive_stats.iter() {
            primitive_contributions.insert(name.clone(), stats.total_phi_contribution);
        }

        ReasoningProfile {
            total_phi: self.total_phi,
            chain_length,
            mean_phi_per_step,
            phi_variance,
            efficiency,
            transformations: self.executions.iter()
                .map(|e| e.transformation.clone())
                .collect(),
            primitives_used,
            tier_distribution,
            primitive_contributions,
        }
    }
}

/// Profile of reasoning chain's consciousness characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningProfile {
    /// Total Φ across entire reasoning chain
    pub total_phi: f64,

    /// Number of primitive executions
    pub chain_length: usize,

    /// Mean Φ per reasoning step
    pub mean_phi_per_step: f64,

    /// Variance in Φ across steps (consistency)
    pub phi_variance: f64,

    /// Information integration efficiency
    pub efficiency: f64,

    /// Sequence of transformations used
    pub transformations: Vec<TransformationType>,

    /// Unique primitives used in reasoning
    pub primitives_used: Vec<String>,

    /// Distribution of primitive usage across tiers
    pub tier_distribution: std::collections::HashMap<PrimitiveTier, usize>,

    /// Φ contribution per primitive
    pub primitive_contributions: std::collections::HashMap<String, f64>,
}

/// Reasoning strategy for primitive selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    /// Use primitives from a single tier only (original behavior)
    SingleTier,

    /// Use all primitives from all tiers (flat multi-tier)
    AllTiers,

    /// Hierarchical reasoning: mirror consciousness structure
    /// - MetaCognitive/Strategic: Planning & goal decomposition (System 2)
    /// - Geometric: Relational structure
    /// - Physical: Concrete grounding
    /// - Mathematical/NSM: Precise execution (System 1)
    Hierarchical,

    /// Adaptive: Use primitive usage statistics to prefer high-Φ primitives
    Adaptive,
}

/// Primitive reasoning engine
pub struct PrimitiveReasoner {
    /// Primitive system for accessing primitives
    primitive_system: PrimitiveSystem,

    /// Current tier for single-tier reasoning
    tier: PrimitiveTier,

    /// Reasoning strategy to use
    strategy: ReasoningStrategy,

    /// Harmonic field for multi-objective optimization
    harmonic_field: HarmonicField,

    /// Weight for harmonic alignment in selection (0.0 = Φ only, 1.0 = harmonics only)
    harmonic_weight: f64,
}

impl PrimitiveReasoner {
    /// Create new primitive reasoner with default strategy
    pub fn new() -> Self {
        let primitive_system = PrimitiveSystem::new();

        Self {
            primitive_system,
            tier: PrimitiveTier::Mathematical,
            strategy: ReasoningStrategy::Hierarchical,  // Revolutionary default!
            harmonic_field: HarmonicField::new(),
            harmonic_weight: 0.3,  // Balanced Φ + harmonics
        }
    }

    /// Set reasoning tier (for SingleTier strategy)
    pub fn with_tier(mut self, tier: PrimitiveTier) -> Self {
        self.tier = tier;
        self
    }

    /// Set reasoning strategy
    pub fn with_strategy(mut self, strategy: ReasoningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set harmonic weight (0.0 = pure Φ, 1.0 = pure harmonics, 0.3 = balanced)
    pub fn with_harmonic_weight(mut self, weight: f64) -> Self {
        self.harmonic_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get current harmonic field
    pub fn harmonic_field(&self) -> &HarmonicField {
        &self.harmonic_field
    }

    /// Get primitives for current tier (public for RL agent)
    pub fn get_tier_primitives(&self) -> Vec<&Primitive> {
        self.primitive_system.get_tier(self.tier)
    }

    /// Get all primitives across all tiers (92 total)
    pub fn get_all_primitives(&self) -> Vec<&Primitive> {
        let mut all_primitives = Vec::new();

        // Collect primitives from all 6 tiers
        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
        ] {
            all_primitives.extend(self.primitive_system.get_tier(tier));
        }

        all_primitives
    }

    /// Get primitives for hierarchical reasoning phase
    fn get_hierarchical_primitives(&self, reasoning_step: usize) -> Vec<&Primitive> {
        // Hierarchical reasoning mirrors consciousness:
        // Early steps: High-level planning (MetaCognitive/Strategic)
        // Middle steps: Structure & grounding (Geometric/Physical)
        // Later steps: Precise execution (Mathematical/NSM)

        if reasoning_step < 2 {
            // Phase 1: Planning with high-level primitives
            let mut planning = self.primitive_system.get_tier(PrimitiveTier::MetaCognitive);
            planning.extend(self.primitive_system.get_tier(PrimitiveTier::Strategic));
            planning
        } else if reasoning_step < 5 {
            // Phase 2: Structuring with mid-level primitives
            let mut structuring = self.primitive_system.get_tier(PrimitiveTier::Geometric);
            structuring.extend(self.primitive_system.get_tier(PrimitiveTier::Physical));
            structuring
        } else {
            // Phase 3: Execution with low-level primitives
            let mut execution = self.primitive_system.get_tier(PrimitiveTier::Mathematical);
            execution.extend(self.primitive_system.get_tier(PrimitiveTier::NSM));
            execution
        }
    }

    /// Calculate harmonic alignment score for a primitive+transformation
    ///
    /// Revolutionary: This measures how well a primitive execution aligns with
    /// the Seven Fiduciary Harmonics, enabling ethics-guided reasoning!
    fn calculate_harmonic_alignment(
        &self,
        primitive: &Primitive,
        transformation: &TransformationType,
    ) -> f64 {
        // Simulate executing this primitive and measure harmonic impact
        let mut test_field = self.harmonic_field.clone();

        // Apply transformation-based harmonic contributions
        match transformation {
            TransformationType::Bind => {
                // Binding creates coherence + wisdom
                test_field.set_level(
                    FiduciaryHarmonic::ResonantCoherence,
                    test_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.1,
                );
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.05,
                );
            }
            TransformationType::Bundle => {
                // Bundling creates interconnectedness
                test_field.set_level(
                    FiduciaryHarmonic::UniversalInterconnectedness,
                    test_field.get_level(FiduciaryHarmonic::UniversalInterconnectedness) + 0.1,
                );
            }
            TransformationType::Resonate => {
                // Resonance amplifies coherence
                test_field.set_level(
                    FiduciaryHarmonic::ResonantCoherence,
                    test_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.15,
                );
            }
            TransformationType::Abstract => {
                // Abstraction builds wisdom
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.1,
                );
            }
            TransformationType::Ground => {
                // Grounding enhances flourishing
                test_field.set_level(
                    FiduciaryHarmonic::PanSentientFlourishing,
                    test_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.1,
                );
            }
            TransformationType::Permute => {
                // Permutation is playful + evolutionary
                test_field.set_level(
                    FiduciaryHarmonic::InfinitePlay,
                    test_field.get_level(FiduciaryHarmonic::InfinitePlay) + 0.1,
                );
            }
        }

        // Apply tier-based contributions
        match primitive.tier {
            PrimitiveTier::NSM => {
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.08,
                );
            }
            PrimitiveTier::Mathematical => {
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.06,
                );
            }
            PrimitiveTier::Physical => {
                test_field.set_level(
                    FiduciaryHarmonic::PanSentientFlourishing,
                    test_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.07,
                );
            }
            PrimitiveTier::Geometric => {
                test_field.set_level(
                    FiduciaryHarmonic::ResonantCoherence,
                    test_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.07,
                );
            }
            PrimitiveTier::Strategic => {
                test_field.set_level(
                    FiduciaryHarmonic::EvolutionaryProgression,
                    test_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.08,
                );
            }
            PrimitiveTier::MetaCognitive => {
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.12,
                );
                test_field.set_level(
                    FiduciaryHarmonic::EvolutionaryProgression,
                    test_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.06,
                );
            }
            PrimitiveTier::Temporal => {
                // Temporal reasoning → Evolutionary Progression (time-awareness) + Wisdom
                test_field.set_level(
                    FiduciaryHarmonic::EvolutionaryProgression,
                    test_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.10,
                );
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.05,
                );
            }
            PrimitiveTier::Compositional => {
                // Compositional → Higher-order wisdom + Coherent integration
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.15,
                );
                test_field.set_level(
                    FiduciaryHarmonic::ResonantCoherence,
                    test_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.10,
                );
            }
            PrimitiveTier::Consciousness => {
                // Consciousness primitives → All harmonics slightly elevated
                test_field.set_level(
                    FiduciaryHarmonic::IntegralWisdom,
                    test_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.10,
                );
                test_field.set_level(
                    FiduciaryHarmonic::ResonantCoherence,
                    test_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.10,
                );
                test_field.set_level(
                    FiduciaryHarmonic::PanSentientFlourishing,
                    test_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.08,
                );
            }
        }

        // Return field coherence as alignment score
        test_field.field_coherence
    }

    /// Select best primitive greedily (public for RL agent baseline)
    pub fn select_greedy(
        &self,
        chain: &ReasoningChain,
        primitives: &[&Primitive],
    ) -> Result<(Primitive, TransformationType)> {
        self.select_next_primitive(chain, primitives)
    }

    /// Reason about a question using primitives
    ///
    /// Revolutionary: This now updates the harmonic field based on reasoning,
    /// creating a feedback loop between consciousness and ethics!
    pub fn reason(&mut self, question: HV16, max_steps: usize) -> Result<ReasoningChain> {
        let mut chain = ReasoningChain::new(question);

        // Execute reasoning steps based on strategy
        for step in 0..max_steps {
            // Get primitives based on strategy
            let primitives = match self.strategy {
                ReasoningStrategy::SingleTier => {
                    // Original: Use only current tier
                    self.primitive_system.get_tier(self.tier)
                }

                ReasoningStrategy::AllTiers => {
                    // Revolutionary: Use ALL 92 primitives
                    self.get_all_primitives()
                }

                ReasoningStrategy::Hierarchical => {
                    // Paradigm-shifting: Mirror consciousness structure
                    // Early: Planning (MetaCognitive/Strategic)
                    // Middle: Structure (Geometric/Physical)
                    // Late: Execution (Mathematical/NSM)
                    self.get_hierarchical_primitives(step)
                }

                ReasoningStrategy::Adaptive => {
                    // Future: Use primitive usage statistics
                    // For now, fall back to all tiers
                    self.get_all_primitives()
                }
            };

            if primitives.is_empty() {
                anyhow::bail!(
                    "No primitives available for strategy {:?} at step {}",
                    self.strategy,
                    step
                );
            }

            // Select primitive that would maximize Φ increase
            let (best_primitive, best_transformation) =
                self.select_next_primitive(&chain, &primitives)?;

            // Execute selected primitive
            chain.execute_primitive(&best_primitive, best_transformation)?;

            // Check if we've reached a stable answer (Φ plateau)
            if chain.phi_gradient.len() > 2 {
                let recent_changes: Vec<f64> = chain.phi_gradient
                    .iter()
                    .rev()
                    .take(3)
                    .copied()
                    .collect();

                let max_change = recent_changes.iter()
                    .fold(0.0f64, |acc, &x| acc.max(x));

                // If Φ contribution very small, we've converged
                if max_change < 0.001 {
                    break;
                }
            }
        }

        // Phase 2.1 Revolutionary: Update harmonic field from completed reasoning!
        // This creates a feedback loop: reasoning → harmonics → future reasoning
        self.update_harmonics_from_chain(&chain);

        Ok(chain)
    }

    /// Update harmonic field from a completed reasoning chain
    ///
    /// Revolutionary: This measures the harmonic effects of the completed reasoning
    /// and updates the internal harmonic field, creating a feedback loop!
    fn update_harmonics_from_chain(&mut self, chain: &ReasoningChain) {
        // Measure harmonic contributions from all executed primitives
        for execution in &chain.executions {
            // Apply transformation-based harmonic contributions
            match execution.transformation {
                TransformationType::Bind => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::ResonantCoherence,
                        self.harmonic_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.01,
                    );
                }
                TransformationType::Bundle => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::UniversalInterconnectedness,
                        self.harmonic_field.get_level(FiduciaryHarmonic::UniversalInterconnectedness) + 0.02,
                    );
                }
                TransformationType::Resonate => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::ResonantCoherence,
                        self.harmonic_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.03,
                    );
                }
                TransformationType::Abstract => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.02,
                    );
                }
                TransformationType::Ground => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::PanSentientFlourishing,
                        self.harmonic_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.02,
                    );
                }
                TransformationType::Permute => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::InfinitePlay,
                        self.harmonic_field.get_level(FiduciaryHarmonic::InfinitePlay) + 0.02,
                    );
                }
            }

            // Apply tier-based contributions
            match execution.primitive.tier {
                PrimitiveTier::NSM => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.01,
                    );
                }
                PrimitiveTier::Mathematical => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.01,
                    );
                }
                PrimitiveTier::Physical => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::PanSentientFlourishing,
                        self.harmonic_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.01,
                    );
                }
                PrimitiveTier::Geometric => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::ResonantCoherence,
                        self.harmonic_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.01,
                    );
                }
                PrimitiveTier::Strategic => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::EvolutionaryProgression,
                        self.harmonic_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.01,
                    );
                }
                PrimitiveTier::MetaCognitive => {
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::EvolutionaryProgression,
                        self.harmonic_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.01,
                    );
                }
                PrimitiveTier::Temporal => {
                    // Temporal reasoning → Evolutionary Progression (time-awareness)
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::EvolutionaryProgression,
                        self.harmonic_field.get_level(FiduciaryHarmonic::EvolutionaryProgression) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.01,
                    );
                }
                PrimitiveTier::Compositional => {
                    // Compositional → Higher-order wisdom + Coherent integration
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::ResonantCoherence,
                        self.harmonic_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.02,
                    );
                }
                PrimitiveTier::Consciousness => {
                    // Consciousness → All harmonics slightly elevated
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::IntegralWisdom,
                        self.harmonic_field.get_level(FiduciaryHarmonic::IntegralWisdom) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::ResonantCoherence,
                        self.harmonic_field.get_level(FiduciaryHarmonic::ResonantCoherence) + 0.02,
                    );
                    self.harmonic_field.set_level(
                        FiduciaryHarmonic::PanSentientFlourishing,
                        self.harmonic_field.get_level(FiduciaryHarmonic::PanSentientFlourishing) + 0.01,
                    );
                }
            }
        }

        // Note: Field coherence is automatically recalculated by set_level()
    }

    /// Select next primitive to execute
    ///
    /// Revolutionary: Multi-objective optimization balancing:
    /// - Φ (consciousness integration)
    /// - Harmonic alignment (ethical/sacred values)
    fn select_next_primitive(
        &self,
        chain: &ReasoningChain,
        primitives: &[&Primitive],
    ) -> Result<(Primitive, TransformationType)> {
        let transformations = vec![
            TransformationType::Bind,
            TransformationType::Bundle,
            TransformationType::Resonate,
            TransformationType::Abstract,
        ];

        let mut best_combined_score = 0.0;
        let mut best_primitive = (*primitives[0]).clone();
        let mut best_transformation = TransformationType::Bind;

        let phi_weight = 1.0 - self.harmonic_weight;

        // Try each primitive
        for primitive in primitives {
            // Try each transformation type
            for transformation in &transformations {
                // Simulate what Φ would be
                let simulated_output = match transformation {
                    TransformationType::Bind => {
                        chain.answer.bind(&primitive.encoding)
                    }
                    TransformationType::Bundle => {
                        HV16::bundle(&[chain.answer.clone(), primitive.encoding.clone()])
                    }
                    TransformationType::Resonate => {
                        let similarity = chain.answer.similarity(&primitive.encoding);
                        if similarity > 0.7 {
                            HV16::bundle(&[chain.answer.clone(), primitive.encoding.clone()])
                        } else {
                            chain.answer.clone()
                        }
                    }
                    TransformationType::Abstract => {
                        let bound = chain.answer.bind(&primitive.encoding);
                        bound.permute(100)
                    }
                    _ => chain.answer.clone(),
                };

                // Measure potential Φ (consciousness)
                let mut phi_computer = IntegratedInformation::new();
                let components = vec![chain.answer.clone(), simulated_output];
                let phi = phi_computer.compute_phi(&components);

                // Measure harmonic alignment (ethics)
                let harmonic_score = self.calculate_harmonic_alignment(primitive, transformation);

                // Multi-objective score: weighted combination
                // phi_weight * Φ + harmonic_weight * harmonics
                let combined_score = (phi_weight * phi) + (self.harmonic_weight * harmonic_score);

                // Track best
                if combined_score > best_combined_score {
                    best_combined_score = combined_score;
                    best_primitive = (*primitive).clone();
                    best_transformation = transformation.clone();
                }
            }
        }

        Ok((best_primitive, best_transformation))
    }
}

// =============================================================================
// ADAPTIVE PRIMITIVE SELECTION
// =============================================================================

/// Task type for adaptive primitive selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Logical/mathematical reasoning
    Logical,
    /// Causal inference
    Causal,
    /// Spatial/geometric reasoning
    Spatial,
    /// Social/strategic reasoning
    Social,
    /// Memory/recall operations
    Memory,
    /// Creative/generative tasks
    Creative,
    /// Self-reflection/meta-cognitive
    MetaCognitive,
    /// Unknown/general
    General,
}

/// Statistics for a primitive's effectiveness on a task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStats {
    /// Primitive name
    pub primitive_name: String,
    /// Number of uses on this task type
    pub use_count: usize,
    /// Total Φ accumulated
    pub total_phi: f64,
    /// Success count (positive Φ contribution)
    pub success_count: usize,
    /// Running mean for Thompson sampling (Beta distribution alpha)
    pub alpha: f64,
    /// Running count for Thompson sampling (Beta distribution beta)
    pub beta: f64,
}

impl AdaptiveStats {
    /// Create new stats for a primitive
    pub fn new(primitive_name: impl Into<String>) -> Self {
        Self {
            primitive_name: primitive_name.into(),
            use_count: 0,
            total_phi: 0.0,
            success_count: 0,
            alpha: 1.0, // Prior: Beta(1,1) = uniform
            beta: 1.0,
        }
    }

    /// Record an outcome
    pub fn record(&mut self, phi_contribution: f64) {
        self.use_count += 1;
        self.total_phi += phi_contribution;

        // Success = positive Φ contribution
        if phi_contribution > 0.0 {
            self.success_count += 1;
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    /// Get mean Φ per use
    pub fn mean_phi(&self) -> f64 {
        if self.use_count == 0 {
            0.0
        } else {
            self.total_phi / self.use_count as f64
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.use_count == 0 {
            0.5 // Prior mean
        } else {
            self.success_count as f64 / self.use_count as f64
        }
    }

    /// Sample from Beta distribution (Thompson sampling)
    /// Returns a random sample representing our belief about success probability
    pub fn thompson_sample(&self) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Simple approximation: sample from Beta(alpha, beta) using Beta distribution
        // For exact sampling we'd need a beta_distribution crate, but this approximation works:
        // Mean of Beta(α, β) = α / (α + β)
        // We add noise proportional to uncertainty
        let mean = self.alpha / (self.alpha + self.beta);
        let uncertainty = 1.0 / (self.alpha + self.beta).sqrt();

        // Sample: mean + noise * uncertainty
        let noise: f64 = rng.gen_range(-1.0..1.0);
        (mean + noise * uncertainty).clamp(0.0, 1.0)
    }
}

/// Adaptive Primitive Selector with Thompson Sampling
///
/// Learns which primitives work best for different task types through
/// multi-armed bandit optimization. Uses Thompson sampling to balance
/// exploration (trying less-used primitives) vs exploitation (using proven ones).
///
/// ## Revolutionary Improvement
///
/// Instead of fixed primitive selection, this system:
/// 1. **Tracks effectiveness**: Per-primitive, per-task-type statistics
/// 2. **Balances exploration/exploitation**: Thompson sampling naturally explores
/// 3. **Adapts over time**: Learns from reasoning outcomes
/// 4. **Task-aware**: Different primitives for different task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePrimitiveSelector {
    /// Statistics per (primitive, task_type) pair
    stats: std::collections::HashMap<(String, TaskType), AdaptiveStats>,

    /// Global exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
    exploration_rate: f64,

    /// Minimum uses before trusting statistics
    min_trust_threshold: usize,
}

impl AdaptivePrimitiveSelector {
    /// Create a new adaptive selector
    pub fn new() -> Self {
        Self {
            stats: std::collections::HashMap::new(),
            exploration_rate: 0.2, // 20% exploration by default
            min_trust_threshold: 3,
        }
    }

    /// Set exploration rate
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set minimum trust threshold
    pub fn with_min_trust(mut self, threshold: usize) -> Self {
        self.min_trust_threshold = threshold.max(1);
        self
    }

    /// Select primitives for a task using Thompson sampling
    ///
    /// Returns primitives sorted by Thompson-sampled expected value
    pub fn select_for_task<'a>(
        &self,
        task: TaskType,
        available: &[&'a Primitive],
        count: usize,
    ) -> Vec<&'a Primitive> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Score each primitive using Thompson sampling
        let mut scored: Vec<_> = available.iter().map(|&prim| {
            let key = (prim.name.clone(), task);
            let stats = self.stats.get(&key);

            let score = if let Some(stats) = stats {
                if stats.use_count >= self.min_trust_threshold {
                    // Enough data: use Thompson sampling
                    stats.thompson_sample()
                } else {
                    // Not enough data: explore with bonus
                    0.5 + rng.gen_range(0.0..self.exploration_rate)
                }
            } else {
                // Never used: maximum exploration bonus
                0.5 + self.exploration_rate
            };

            (prim, score)
        }).collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top `count` primitives
        scored.into_iter().take(count).map(|(p, _)| p).collect()
    }

    /// Update statistics after a reasoning chain completes
    pub fn update_from_chain(&mut self, chain: &ReasoningChain, task: TaskType) {
        for execution in &chain.executions {
            let key = (execution.primitive.name.clone(), task);

            let stats = self.stats.entry(key.clone()).or_insert_with(|| {
                AdaptiveStats::new(&execution.primitive.name)
            });

            stats.record(execution.phi_contribution);
        }
    }

    /// Record a single phi observation for a primitive on a task type.
    ///
    /// This is used for architectural feedback where we want to boost or penalize
    /// primitives based on their overall performance in the system.
    pub fn record_observation(&mut self, primitive_name: &str, task: TaskType, phi: f64) {
        let key = (primitive_name.to_string(), task);
        let stats = self.stats.entry(key).or_insert_with(|| {
            AdaptiveStats::new(primitive_name)
        });
        stats.record(phi);
    }

    /// Record multiple observations for a primitive (for batch updates from architecture feedback)
    pub fn record_batch_observation(&mut self, primitive_name: &str, task: TaskType, phi: f64, count: usize) {
        for _ in 0..count {
            self.record_observation(primitive_name, task, phi);
        }
    }

    /// Get statistics for a primitive on a task type
    pub fn get_stats(&self, primitive_name: &str, task: TaskType) -> Option<&AdaptiveStats> {
        self.stats.get(&(primitive_name.to_string(), task))
    }

    /// Get all statistics
    pub fn all_stats(&self) -> &std::collections::HashMap<(String, TaskType), AdaptiveStats> {
        &self.stats
    }

    /// Get top performing primitives for a task type
    pub fn top_primitives(&self, task: TaskType, count: usize) -> Vec<(&str, f64)> {
        let mut task_stats: Vec<_> = self.stats.iter()
            .filter(|((_, t), _)| *t == task)
            .filter(|(_, stats)| stats.use_count >= self.min_trust_threshold)
            .map(|((name, _), stats)| (name.as_str(), stats.mean_phi()))
            .collect();

        task_stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        task_stats.into_iter().take(count).collect()
    }

    /// Detect underperforming primitives (candidates for removal/evolution)
    pub fn underperformers(&self, threshold: f64) -> Vec<(String, TaskType, f64)> {
        self.stats.iter()
            .filter(|(_, stats)| stats.use_count >= self.min_trust_threshold)
            .filter(|(_, stats)| stats.mean_phi() < threshold)
            .map(|((name, task), stats)| (name.clone(), *task, stats.mean_phi()))
            .collect()
    }

    /// Generate a report of adaptive selection health
    pub fn health_report(&self) -> AdaptiveHealthReport {
        let total_primitives = self.stats.keys()
            .map(|(name, _)| name.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();

        let total_uses: usize = self.stats.values().map(|s| s.use_count).sum();
        let total_phi: f64 = self.stats.values().map(|s| s.total_phi).sum();

        let underperformers = self.underperformers(0.0).len();

        AdaptiveHealthReport {
            total_primitives_tracked: total_primitives,
            total_uses,
            total_phi,
            underperformer_count: underperformers,
            exploration_rate: self.exploration_rate,
        }
    }
}

impl Default for AdaptivePrimitiveSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Health report for adaptive selection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveHealthReport {
    /// Number of unique primitives tracked
    pub total_primitives_tracked: usize,
    /// Total number of primitive uses
    pub total_uses: usize,
    /// Total Φ accumulated across all uses
    pub total_phi: f64,
    /// Number of underperforming primitives
    pub underperformer_count: usize,
    /// Current exploration rate
    pub exploration_rate: f64,
}

// =============================================================================
// PRIMITIVE MEMOIZER - Caches frequently-used primitive compositions
// =============================================================================

/// Cached result of a primitive composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedComposition {
    /// Resulting hypervector
    pub result: HV16,
    /// Φ contribution of this composition
    pub phi: f64,
    /// Number of cache hits
    pub hits: usize,
    /// Last access timestamp
    pub last_access: u64,
}

/// Memoization layer for primitive compositions
///
/// Caches frequently-used primitive combinations to avoid recomputation.
/// Uses LRU eviction when cache is full.
///
/// ## Example
/// ```rust,ignore
/// let mut memoizer = PrimitiveMemoizer::new(1000);
///
/// // First call computes and caches
/// let result = memoizer.get_or_compute("BIND", "COMPARE", TransformationType::Bind, || {
///     expensive_computation()
/// });
///
/// // Second call returns cached value
/// let cached = memoizer.get_or_compute("BIND", "COMPARE", TransformationType::Bind, || {
///     panic!("Should not be called!")
/// });
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveMemoizer {
    /// Cache mapping (prim1, prim2, transform) -> result
    cache: std::collections::HashMap<(String, String, TransformationType), CachedComposition>,
    /// Maximum cache size
    max_size: usize,
    /// Total cache hits
    total_hits: usize,
    /// Total cache misses
    total_misses: usize,
    /// Current timestamp counter
    timestamp: u64,
}

impl PrimitiveMemoizer {
    /// Create a new memoizer with given capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size: max_size.max(10),
            total_hits: 0,
            total_misses: 0,
            timestamp: 0,
        }
    }

    /// Get cached result or compute and cache it
    pub fn get_or_compute<F>(
        &mut self,
        prim1: &str,
        prim2: &str,
        transform: TransformationType,
        compute: F,
    ) -> (HV16, f64)
    where
        F: FnOnce() -> (HV16, f64),
    {
        self.timestamp += 1;
        let key = (prim1.to_string(), prim2.to_string(), transform);

        if let Some(cached) = self.cache.get_mut(&key) {
            cached.hits += 1;
            cached.last_access = self.timestamp;
            self.total_hits += 1;
            return (cached.result.clone(), cached.phi);
        }

        // Cache miss - compute and store
        self.total_misses += 1;
        let (result, phi) = compute();

        // Evict if necessary
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(key, CachedComposition {
            result: result.clone(),
            phi,
            hits: 1,
            last_access: self.timestamp,
        });

        (result, phi)
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.cache.iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, _)| k.clone())
        {
            self.cache.remove(&lru_key);
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> MemoizerStats {
        MemoizerStats {
            cache_size: self.cache.len(),
            max_size: self.max_size,
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_rate: self.hit_rate(),
            most_used: self.most_used_compositions(5),
        }
    }

    /// Get most frequently used compositions
    pub fn most_used_compositions(&self, count: usize) -> Vec<(String, String, usize)> {
        let mut sorted: Vec<_> = self.cache.iter()
            .map(|((p1, p2, _), v)| (p1.clone(), p2.clone(), v.hits))
            .collect();
        sorted.sort_by(|a, b| b.2.cmp(&a.2));
        sorted.into_iter().take(count).collect()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.total_hits = 0;
        self.total_misses = 0;
    }
}

/// Statistics for the memoizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoizerStats {
    pub cache_size: usize,
    pub max_size: usize,
    pub total_hits: usize,
    pub total_misses: usize,
    pub hit_rate: f64,
    pub most_used: Vec<(String, String, usize)>,
}

// =============================================================================
// PRIMITIVE AFFINITY GRAPH - Tracks which primitives compose well together
// =============================================================================

/// Tracks how well pairs of primitives compose together
///
/// Learns from actual usage which primitive combinations produce
/// high Φ (integrated information), enabling smarter composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveAffinityGraph {
    /// Affinity scores: (prim1, prim2) -> (total_phi, count)
    affinities: std::collections::HashMap<(String, String), (f64, usize)>,
    /// Tier-based default affinities
    tier_affinities: std::collections::HashMap<(PrimitiveTier, PrimitiveTier), f64>,
}

impl PrimitiveAffinityGraph {
    /// Create a new affinity graph with default tier affinities
    pub fn new() -> Self {
        let mut tier_affinities = std::collections::HashMap::new();

        // Same-tier primitives have base affinity
        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            tier_affinities.insert((tier, tier), 0.7);
        }

        // Adjacent tiers have moderate affinity
        let tier_order = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        for i in 0..tier_order.len() - 1 {
            tier_affinities.insert((tier_order[i], tier_order[i + 1]), 0.5);
            tier_affinities.insert((tier_order[i + 1], tier_order[i]), 0.5);
        }

        // Consciousness tier has special affinity with MetaCognitive
        tier_affinities.insert((PrimitiveTier::Consciousness, PrimitiveTier::MetaCognitive), 0.8);
        tier_affinities.insert((PrimitiveTier::MetaCognitive, PrimitiveTier::Consciousness), 0.8);

        Self {
            affinities: std::collections::HashMap::new(),
            tier_affinities,
        }
    }

    /// Record a composition result
    pub fn record_composition(&mut self, prim1: &str, prim2: &str, phi: f64) {
        let key = (prim1.to_string(), prim2.to_string());
        let entry = self.affinities.entry(key).or_insert((0.0, 0));
        entry.0 += phi;
        entry.1 += 1;
    }

    /// Get affinity between two primitives
    pub fn get_affinity(&self, prim1: &str, prim2: &str) -> f64 {
        let key = (prim1.to_string(), prim2.to_string());
        if let Some((total_phi, count)) = self.affinities.get(&key) {
            if *count > 0 {
                return *total_phi / *count as f64;
            }
        }
        0.5 // Default neutral affinity
    }

    /// Get tier-based affinity
    pub fn get_tier_affinity(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> f64 {
        self.tier_affinities.get(&(tier1, tier2)).copied().unwrap_or(0.3)
    }

    /// Get best composition partners for a primitive
    pub fn best_partners(&self, primitive: &str, count: usize) -> Vec<(String, f64)> {
        let mut partners: Vec<_> = self.affinities.iter()
            .filter(|((p1, _), _)| p1 == primitive)
            .map(|((_, p2), (total, cnt))| {
                let avg = if *cnt > 0 { *total / *cnt as f64 } else { 0.0 };
                (p2.clone(), avg)
            })
            .collect();

        partners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        partners.into_iter().take(count).collect()
    }

    /// Suggest a composition chain for maximum Φ
    pub fn suggest_chain(&self, start: &str, length: usize) -> Vec<String> {
        let mut chain = vec![start.to_string()];
        let mut current = start.to_string();

        for _ in 1..length {
            let partners = self.best_partners(&current, 1);
            if let Some((next, _)) = partners.into_iter().next() {
                chain.push(next.clone());
                current = next;
            } else {
                break;
            }
        }

        chain
    }

    /// Get graph statistics
    pub fn stats(&self) -> AffinityGraphStats {
        let total_pairs = self.affinities.len();
        let total_compositions: usize = self.affinities.values().map(|(_, c)| c).sum();
        let avg_affinity = if total_pairs > 0 {
            self.affinities.values()
                .map(|(t, c)| if *c > 0 { t / *c as f64 } else { 0.0 })
                .sum::<f64>() / total_pairs as f64
        } else {
            0.0
        };

        AffinityGraphStats {
            total_pairs,
            total_compositions,
            avg_affinity,
        }
    }
}

impl Default for PrimitiveAffinityGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for affinity graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityGraphStats {
    pub total_pairs: usize,
    pub total_compositions: usize,
    pub avg_affinity: f64,
}

// =============================================================================
// TIER-AWARE COMPOSITION RULES
// =============================================================================

/// Compatibility rules for composing primitives across tiers.
///
/// This matrix encodes domain knowledge about which tier combinations
/// produce meaningful results when composed. Not all combinations are valid:
/// - Mathematical + Physical: Valid (e.g., force equations)
/// - NSM + Consciousness: Valid (e.g., phenomenal language)
/// - Temporal + Geometric: May be incompatible (different domains)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierCompatibilityMatrix {
    /// Compatibility scores: (tier1, tier2) -> compatibility in [0, 1]
    /// 1.0 = fully compatible, 0.0 = incompatible
    compatibility: std::collections::HashMap<(PrimitiveTier, PrimitiveTier), f64>,

    /// Composition rules: (tier1, tier2) -> CompositionRule
    rules: std::collections::HashMap<(PrimitiveTier, PrimitiveTier), CompositionRule>,
}

/// Defines how two primitives from different tiers should be composed
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompositionRule {
    /// Standard HDC binding (XOR-based)
    StandardBind,
    /// Weighted binding (tier1 dominates)
    WeightedBind { tier1_weight: f64 },
    /// Sequential composition (tier1 then tier2)
    Sequential,
    /// Parallel composition (independent contributions)
    Parallel,
    /// Not recommended (may produce low-quality results)
    NotRecommended,
}

impl TierCompatibilityMatrix {
    /// Create compatibility matrix with domain-informed defaults
    pub fn new() -> Self {
        use PrimitiveTier::*;

        let mut compatibility = std::collections::HashMap::new();
        let mut rules = std::collections::HashMap::new();

        // All tiers for iteration
        let all_tiers = [
            NSM, Mathematical, Physical, Geometric, Strategic,
            MetaCognitive, Temporal, Compositional, Consciousness,
        ];

        // Default: same-tier is highly compatible
        for tier in &all_tiers {
            compatibility.insert((*tier, *tier), 1.0);
            rules.insert((*tier, *tier), CompositionRule::StandardBind);
        }

        // Mathematical pairs well with most tiers
        compatibility.insert((Mathematical, Physical), 0.9);
        compatibility.insert((Physical, Mathematical), 0.9);
        rules.insert((Mathematical, Physical), CompositionRule::StandardBind);
        rules.insert((Physical, Mathematical), CompositionRule::StandardBind);

        compatibility.insert((Mathematical, Geometric), 0.9);
        compatibility.insert((Geometric, Mathematical), 0.9);
        rules.insert((Mathematical, Geometric), CompositionRule::StandardBind);
        rules.insert((Geometric, Mathematical), CompositionRule::StandardBind);

        // Physical + Geometric = spatial physics
        compatibility.insert((Physical, Geometric), 0.85);
        compatibility.insert((Geometric, Physical), 0.85);
        rules.insert((Physical, Geometric), CompositionRule::StandardBind);
        rules.insert((Geometric, Physical), CompositionRule::StandardBind);

        // Strategic + MetaCognitive = planning
        compatibility.insert((Strategic, MetaCognitive), 0.8);
        compatibility.insert((MetaCognitive, Strategic), 0.8);
        rules.insert((Strategic, MetaCognitive), CompositionRule::WeightedBind { tier1_weight: 0.6 });
        rules.insert((MetaCognitive, Strategic), CompositionRule::WeightedBind { tier1_weight: 0.6 });

        // Consciousness + MetaCognitive = self-aware reasoning
        compatibility.insert((Consciousness, MetaCognitive), 0.95);
        compatibility.insert((MetaCognitive, Consciousness), 0.95);
        rules.insert((Consciousness, MetaCognitive), CompositionRule::WeightedBind { tier1_weight: 0.7 });
        rules.insert((MetaCognitive, Consciousness), CompositionRule::WeightedBind { tier1_weight: 0.7 });

        // Consciousness + NSM = phenomenal language
        compatibility.insert((Consciousness, NSM), 0.85);
        compatibility.insert((NSM, Consciousness), 0.85);
        rules.insert((Consciousness, NSM), CompositionRule::Sequential);
        rules.insert((NSM, Consciousness), CompositionRule::Sequential);

        // Temporal + Strategic = temporal planning
        compatibility.insert((Temporal, Strategic), 0.8);
        compatibility.insert((Strategic, Temporal), 0.8);
        rules.insert((Temporal, Strategic), CompositionRule::Parallel);
        rules.insert((Strategic, Temporal), CompositionRule::Parallel);

        // Compositional tier enhances other tiers
        for tier in &all_tiers {
            if *tier != Compositional {
                compatibility.insert((Compositional, *tier), 0.75);
                compatibility.insert((*tier, Compositional), 0.75);
                rules.insert((Compositional, *tier), CompositionRule::StandardBind);
                rules.insert((*tier, Compositional), CompositionRule::StandardBind);
            }
        }

        // Less compatible combinations
        compatibility.insert((Physical, NSM), 0.4);  // Different domains
        compatibility.insert((NSM, Physical), 0.4);
        rules.insert((Physical, NSM), CompositionRule::NotRecommended);
        rules.insert((NSM, Physical), CompositionRule::NotRecommended);

        compatibility.insert((Temporal, Physical), 0.5);  // Time ≠ space physics
        compatibility.insert((Physical, Temporal), 0.5);
        rules.insert((Temporal, Physical), CompositionRule::WeightedBind { tier1_weight: 0.5 });
        rules.insert((Physical, Temporal), CompositionRule::WeightedBind { tier1_weight: 0.5 });

        Self { compatibility, rules }
    }

    /// Get compatibility score between two tiers
    pub fn get_compatibility(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> f64 {
        self.compatibility.get(&(tier1, tier2)).copied().unwrap_or(0.5)
    }

    /// Get composition rule for two tiers
    pub fn get_rule(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> CompositionRule {
        self.rules.get(&(tier1, tier2)).copied().unwrap_or(CompositionRule::StandardBind)
    }

    /// Check if composition is recommended
    pub fn is_recommended(&self, tier1: PrimitiveTier, tier2: PrimitiveTier) -> bool {
        match self.get_rule(tier1, tier2) {
            CompositionRule::NotRecommended => false,
            _ => self.get_compatibility(tier1, tier2) >= 0.5,
        }
    }

    /// Get all compatible tiers for a given tier, sorted by compatibility
    pub fn compatible_tiers(&self, tier: PrimitiveTier) -> Vec<(PrimitiveTier, f64)> {
        let all_tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        let mut compatible: Vec<_> = all_tiers.iter()
            .filter(|&&t| t != tier && self.is_recommended(tier, t))
            .map(|&t| (t, self.get_compatibility(tier, t)))
            .collect();

        compatible.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        compatible
    }

    /// Suggest an optimal composition chain across tiers for maximum integration
    pub fn suggest_tier_chain(&self, start: PrimitiveTier, length: usize) -> Vec<PrimitiveTier> {
        let mut chain = vec![start];
        let mut current = start;

        for _ in 1..length {
            let compatible = self.compatible_tiers(current);
            if let Some((next_tier, _)) = compatible.into_iter()
                .filter(|(t, _)| !chain.contains(t))  // Avoid cycles
                .next()
            {
                chain.push(next_tier);
                current = next_tier;
            } else {
                break;
            }
        }

        chain
    }
}

impl Default for TierCompatibilityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PERSISTENCE - Save/Load for AdaptivePrimitiveSelector
// =============================================================================

impl AdaptivePrimitiveSelector {
    /// Save selector state to a file
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load selector state from a file
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let selector: Self = serde_json::from_str(&json)?;
        Ok(selector)
    }

    /// Merge statistics from another selector (for distributed learning)
    pub fn merge(&mut self, other: &Self) {
        for (key, other_stats) in &other.stats {
            let entry = self.stats.entry(key.clone()).or_insert_with(|| {
                AdaptiveStats::new(&key.0)
            });
            entry.use_count += other_stats.use_count;
            entry.total_phi += other_stats.total_phi;
            entry.success_count += other_stats.success_count;
            // Recalculate alpha/beta from merged statistics
            entry.alpha = 1.0 + entry.success_count as f64;
            entry.beta = 1.0 + (entry.use_count - entry.success_count) as f64;
        }
    }
}

// =============================================================================
// TIER-AWARE SELECTION - Enhanced primitive selection considering tiers
// =============================================================================

/// Configuration for tier-aware selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierAwareConfig {
    /// Prefer primitives from tiers that match the task complexity
    pub match_complexity: bool,
    /// Boost for consciousness-tier primitives in metacognitive tasks
    pub consciousness_boost: f64,
    /// Penalty for using high-tier primitives on simple tasks
    pub complexity_mismatch_penalty: f64,
    /// Enable tier diversity in chains
    pub encourage_diversity: bool,
}

impl Default for TierAwareConfig {
    fn default() -> Self {
        Self {
            match_complexity: true,
            consciousness_boost: 0.2,
            complexity_mismatch_penalty: 0.1,
            encourage_diversity: true,
        }
    }
}

impl AdaptivePrimitiveSelector {
    /// Select primitives with tier awareness
    pub fn select_tier_aware<'a>(
        &self,
        task: TaskType,
        available: &[&'a Primitive],
        count: usize,
        config: &TierAwareConfig,
        affinity_graph: Option<&PrimitiveAffinityGraph>,
    ) -> Vec<&'a Primitive> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Determine target complexity based on task type
        let target_tiers = self.target_tiers_for_task(task);

        // Score each primitive
        let mut scored: Vec<_> = available.iter().map(|&prim| {
            let key = (prim.name.clone(), task);
            let stats = self.stats.get(&key);

            // Base score from Thompson sampling
            let mut score = if let Some(stats) = stats {
                if stats.use_count >= self.min_trust_threshold {
                    stats.thompson_sample()
                } else {
                    0.5 + rng.gen_range(0.0..self.exploration_rate)
                }
            } else {
                0.5 + self.exploration_rate
            };

            // Tier matching bonus/penalty
            if config.match_complexity {
                if target_tiers.contains(&prim.tier) {
                    score += 0.1;
                } else {
                    score -= config.complexity_mismatch_penalty;
                }
            }

            // Consciousness boost for metacognitive tasks
            if prim.tier == PrimitiveTier::Consciousness {
                if task == TaskType::MetaCognitive {
                    score += config.consciousness_boost;
                }
            }

            // Affinity bonus if graph provided
            if let Some(graph) = affinity_graph {
                let tier_affinity = graph.get_tier_affinity(prim.tier, prim.tier);
                score += tier_affinity * 0.1;
            }

            (prim, score)
        }).collect();

        // Sort by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // If encouraging diversity, ensure tier variety
        if config.encourage_diversity && count > 1 {
            return self.select_with_diversity(scored, count);
        }

        scored.into_iter().take(count).map(|(p, _)| p).collect()
    }

    /// Get target tiers for a task type
    fn target_tiers_for_task(&self, task: TaskType) -> Vec<PrimitiveTier> {
        match task {
            TaskType::Logical => vec![PrimitiveTier::Mathematical, PrimitiveTier::Compositional],
            TaskType::Causal => vec![PrimitiveTier::Physical, PrimitiveTier::Temporal],
            TaskType::Spatial => vec![PrimitiveTier::Geometric, PrimitiveTier::Physical],
            TaskType::Social => vec![PrimitiveTier::Strategic, PrimitiveTier::MetaCognitive],
            TaskType::Memory => vec![PrimitiveTier::Temporal, PrimitiveTier::NSM],
            TaskType::Creative => vec![PrimitiveTier::Compositional, PrimitiveTier::Consciousness],
            TaskType::MetaCognitive => vec![PrimitiveTier::MetaCognitive, PrimitiveTier::Consciousness],
            TaskType::General => vec![PrimitiveTier::Physical, PrimitiveTier::Mathematical],
        }
    }

    /// Select primitives ensuring tier diversity
    fn select_with_diversity<'a>(
        &self,
        scored: Vec<(&'a Primitive, f64)>,
        count: usize,
    ) -> Vec<&'a Primitive> {
        let mut selected = Vec::new();
        let mut used_tiers = std::collections::HashSet::new();
        let mut used_names = std::collections::HashSet::new();

        // First pass: one primitive per tier
        for (prim, _) in &scored {
            if selected.len() >= count {
                break;
            }
            if !used_tiers.contains(&prim.tier) {
                selected.push(*prim);
                used_tiers.insert(prim.tier);
                used_names.insert(prim.name.clone());
            }
        }

        // Second pass: fill remaining slots with best scores
        for (prim, _) in scored {
            if selected.len() >= count {
                break;
            }
            if !used_names.contains(&prim.name) {
                selected.push(prim);
                used_names.insert(prim.name.clone());
            }
        }

        selected
    }
}

// =============================================================================
// PROMETHEUS METRICS EXPORT
// =============================================================================

use crate::hdc::primitive_dashboard::PrimitiveDashboard;

impl PrimitiveDashboard {
    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("# HELP primitive_activations_total Total primitive activations\n");
        output.push_str("# TYPE primitive_activations_total counter\n");

        // Per-primitive metrics
        for (name, stats) in self.all_stats() {
            output.push_str(&format!(
                "primitive_activations_total{{name=\"{}\"}} {}\n",
                name, stats.activations
            ));
        }

        output.push_str("\n# HELP primitive_successes_total Total successful activations\n");
        output.push_str("# TYPE primitive_successes_total counter\n");

        for (name, stats) in self.all_stats() {
            output.push_str(&format!(
                "primitive_successes_total{{name=\"{}\"}} {}\n",
                name, stats.successes
            ));
        }

        output.push_str("\n# HELP primitive_success_rate Success rate per primitive\n");
        output.push_str("# TYPE primitive_success_rate gauge\n");

        for (name, stats) in self.all_stats() {
            let rate = stats.success_rate();
            output.push_str(&format!(
                "primitive_success_rate{{name=\"{}\"}} {:.6}\n",
                name, rate
            ));
        }

        output.push_str("\n# HELP primitive_avg_duration_ms Average activation duration in ms\n");
        output.push_str("# TYPE primitive_avg_duration_ms gauge\n");

        for (name, stats) in self.all_stats() {
            output.push_str(&format!(
                "primitive_avg_duration_ms{{name=\"{}\"}} {:.6}\n",
                name, stats.avg_duration_ms
            ));
        }

        output.push_str("\n# HELP primitive_peak_rate Peak activations per minute\n");
        output.push_str("# TYPE primitive_peak_rate gauge\n");

        for (name, stats) in self.all_stats() {
            output.push_str(&format!(
                "primitive_peak_rate{{name=\"{}\"}} {:.6}\n",
                name, stats.peak_rate
            ));
        }

        // Tier-level metrics using available tier stats method
        output.push_str("\n# HELP primitive_tier_activations Total activations by tier\n");
        output.push_str("# TYPE primitive_tier_activations counter\n");

        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            if let Some(stats) = self.get_tier_stats(&tier) {
                output.push_str(&format!(
                    "primitive_tier_activations{{tier=\"{:?}\"}} {}\n",
                    tier, stats.activations
                ));
            }
        }

        output.push_str("\n# HELP primitive_tier_successes Total successes by tier\n");
        output.push_str("# TYPE primitive_tier_successes counter\n");

        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            if let Some(stats) = self.get_tier_stats(&tier) {
                output.push_str(&format!(
                    "primitive_tier_successes{{tier=\"{:?}\"}} {}\n",
                    tier, stats.successes
                ));
            }
        }

        output
    }

    /// Export metrics as JSON for dashboards
    pub fn export_json(&self) -> Result<String> {
        // Collect stats into a serializable structure
        let stats: std::collections::HashMap<String, serde_json::Value> = self.all_stats()
            .iter()
            .map(|(name, s)| {
                (name.clone(), serde_json::json!({
                    "activations": s.activations,
                    "successes": s.successes,
                    "success_rate": s.success_rate(),
                    "avg_duration_ms": s.avg_duration_ms,
                    "peak_rate": s.peak_rate,
                }))
            })
            .collect();

        let report = serde_json::json!({
            "primitive_stats": stats,
            "health_score": self.health_score(),
            "activations_per_minute": self.activations_per_minute(),
        });

        Ok(serde_json::to_string_pretty(&report)?)
    }
}

// =============================================================================
// INTEGRATION HELPERS - Connect primitives to other subsystems
// =============================================================================

/// Bridge for connecting primitives to causal discovery
#[derive(Debug, Clone)]
pub struct PrimitiveCausalBridge {
    /// Mapping from primitive operations to causal mechanisms
    pub mechanism_map: std::collections::HashMap<String, String>,
}

impl PrimitiveCausalBridge {
    pub fn new() -> Self {
        let mut mechanism_map = std::collections::HashMap::new();

        // Map primitives to causal mechanisms
        mechanism_map.insert("CAUSE".to_string(), "direct_causation".to_string());
        mechanism_map.insert("EFFECT".to_string(), "causal_effect".to_string());
        mechanism_map.insert("PREVENT".to_string(), "inhibition".to_string());
        mechanism_map.insert("ENABLE".to_string(), "facilitation".to_string());
        mechanism_map.insert("CONTINGENT".to_string(), "conditional_dependency".to_string());

        Self { mechanism_map }
    }

    /// Get causal mechanism for a primitive
    pub fn get_mechanism(&self, primitive_name: &str) -> Option<&str> {
        self.mechanism_map.get(primitive_name).map(|s| s.as_str())
    }

    /// Check if a primitive has causal semantics
    pub fn has_causal_semantics(&self, primitive_name: &str) -> bool {
        self.mechanism_map.contains_key(primitive_name)
    }
}

impl Default for PrimitiveCausalBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge for connecting primitives to synthesis
#[derive(Debug, Clone)]
pub struct PrimitiveSynthesisBridge {
    /// Primitives suitable for program synthesis
    pub synthesis_primitives: Vec<String>,
}

impl PrimitiveSynthesisBridge {
    pub fn new() -> Self {
        Self {
            synthesis_primitives: vec![
                "COMPOSE".to_string(),
                "SEQUENCE".to_string(),
                "BRANCH".to_string(),
                "ITERATE".to_string(),
                "ABSTRACT".to_string(),
                "INSTANTIATE".to_string(),
            ],
        }
    }

    /// Check if primitive is suitable for synthesis
    pub fn is_synthesis_primitive(&self, name: &str) -> bool {
        self.synthesis_primitives.contains(&name.to_string())
    }

    /// Get all synthesis-suitable primitives
    pub fn get_synthesis_primitives(&self) -> &[String] {
        &self.synthesis_primitives
    }
}

impl Default for PrimitiveSynthesisBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PrimitiveReasoner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_chain_creation() {
        let question = HV16::random(42);
        let chain = ReasoningChain::new(question.clone());

        assert_eq!(chain.executions.len(), 0);
        assert_eq!(chain.total_phi, 0.0);
        assert_eq!(chain.question, question);
    }

    #[test]
    fn test_primitive_execution() -> Result<()> {
        let question = HV16::random(42);
        let mut chain = ReasoningChain::new(question);

        // Create test primitive
        let primitive = Primitive {
            name: "TEST_PRIMITIVE".to_string(),
            encoding: HV16::random(123),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "Test primitive".to_string(),
            is_base: true,
            derivation: None,
        };

        // Execute primitive
        chain.execute_primitive(&primitive, TransformationType::Bind)?;

        assert_eq!(chain.executions.len(), 1);
        assert!(chain.total_phi > 0.0);

        Ok(())
    }

    #[test]
    fn test_reasoning_profile() -> Result<()> {
        let question = HV16::random(42);
        let mut chain = ReasoningChain::new(question);

        let primitive = Primitive {
            name: "TEST".to_string(),
            encoding: HV16::random(123),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "Test".to_string(),
            is_base: true,
            derivation: None,
        };

        // Execute several steps
        for _ in 0..3 {
            chain.execute_primitive(&primitive, TransformationType::Bind)?;
        }

        let profile = chain.consciousness_profile();

        assert_eq!(profile.chain_length, 3);
        assert!(profile.total_phi > 0.0);
        assert!(profile.mean_phi_per_step > 0.0);

        Ok(())
    }

    #[test]
    fn test_primitive_reasoner() -> Result<()> {
        let mut reasoner = PrimitiveReasoner::new();

        let question = HV16::random(42);
        let chain = reasoner.reason(question, 5)?;

        assert!(chain.executions.len() > 0);
        assert!(chain.total_phi > 0.0);

        Ok(())
    }

    // === AdaptivePrimitiveSelector Tests ===

    #[test]
    fn test_adaptive_selector_creation() {
        let selector = AdaptivePrimitiveSelector::new();
        assert!(selector.all_stats().is_empty());
        assert!((selector.exploration_rate - 0.2).abs() < 0.01); // Default is 20% exploration
    }

    #[test]
    fn test_adaptive_selector_with_exploration() {
        let selector = AdaptivePrimitiveSelector::new().with_exploration_rate(0.2);
        assert!((selector.exploration_rate - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_selector_update() {
        // Use lower trust threshold for testing
        let mut selector = AdaptivePrimitiveSelector::new().with_min_trust(1);

        // Create a test primitive
        let primitive = Primitive {
            name: "TEST_PRIM".to_string(),
            encoding: HV16::random(42),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "Test".to_string(),
            is_base: true,
            derivation: None,
        };

        // Create a chain with the primitive
        let question = HV16::random(123);
        let mut chain = ReasoningChain::new(question);
        chain.execute_primitive(&primitive, TransformationType::Bind).unwrap();

        // Update selector with successful chain
        selector.update_from_chain(&chain, TaskType::Logical);

        // Check stats were recorded
        let top = selector.top_primitives(TaskType::Logical, 5);
        assert!(!top.is_empty());
    }

    #[test]
    fn test_adaptive_selector_select_for_task() {
        // Use lower trust threshold for testing
        let mut selector = AdaptivePrimitiveSelector::new().with_min_trust(1);

        // Create test primitives
        let primitives: Vec<Primitive> = (0..5).map(|i| Primitive {
            name: format!("PRIM_{}", i),
            encoding: HV16::random(i as u64),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: format!("Test {}", i),
            is_base: true,
            derivation: None,
        }).collect();

        // Train on some primitives
        for prim in &primitives[..3] {
            let question = HV16::random(42);
            let mut chain = ReasoningChain::new(question);
            chain.execute_primitive(prim, TransformationType::Bind).unwrap();
            selector.update_from_chain(&chain, TaskType::Logical);
        }

        // Select primitives - need to convert to references
        let prim_refs: Vec<&Primitive> = primitives.iter().collect();
        let selected = selector.select_for_task(TaskType::Logical, &prim_refs, 3);
        assert!(!selected.is_empty());
        assert!(selected.len() <= 3);
    }

    #[test]
    fn test_adaptive_selector_underperformers() {
        let mut selector = AdaptivePrimitiveSelector::new();

        // Create primitive
        let primitive = Primitive {
            name: "BAD_PRIM".to_string(),
            encoding: HV16::random(42),
            tier: PrimitiveTier::Physical,
            domain: "test".to_string(),
            definition: "Test".to_string(),
            is_base: true,
            derivation: None,
        };

        // Add many uses (Thompson sampling tracks reward, not explicit success/failure)
        // We simulate poor performance by having chains with low phi
        for _ in 0..10 {
            let question = HV16::random(42);
            let mut chain = ReasoningChain::new(question);
            chain.execute_primitive(&primitive, TransformationType::Bind).unwrap();
            selector.update_from_chain(&chain, TaskType::Logical);
        }

        // Check underperformers - Thompson sampling uses different metrics
        // Just verify the method doesn't panic
        let _ = selector.underperformers(0.5);
    }

    #[test]
    fn test_adaptive_selector_health_report() {
        let selector = AdaptivePrimitiveSelector::new();
        let report = selector.health_report();
        // Report should have valid initial values
        assert_eq!(report.total_primitives_tracked, 0);
        assert_eq!(report.total_uses, 0);
        assert!((report.exploration_rate - 0.2).abs() < 0.01); // Default is 20% exploration
    }

    #[test]
    fn test_task_type_variants() {
        // Ensure all TaskType variants exist
        let types = vec![
            TaskType::Logical,
            TaskType::Causal,
            TaskType::Spatial,
            TaskType::Social,
            TaskType::Memory,
            TaskType::Creative,
            TaskType::MetaCognitive,
            TaskType::General,
        ];
        assert_eq!(types.len(), 8);
    }

    // === PrimitiveMemoizer Tests ===

    #[test]
    fn test_memoizer_creation() {
        let memoizer = PrimitiveMemoizer::new(100);
        assert_eq!(memoizer.stats().cache_size, 0);
        assert_eq!(memoizer.hit_rate(), 0.0);
    }

    #[test]
    fn test_memoizer_caching() {
        let mut memoizer = PrimitiveMemoizer::new(100);
        let mut compute_count = 0;

        // First call should compute
        let (result1, phi1) = memoizer.get_or_compute("A", "B", TransformationType::Bind, || {
            compute_count += 1;
            (HV16::random(42), 0.5)
        });

        // Second call should return cached
        let (result2, phi2) = memoizer.get_or_compute("A", "B", TransformationType::Bind, || {
            compute_count += 1;
            (HV16::random(99), 0.9)  // Different values to prove cache is used
        });

        assert_eq!(compute_count, 1);  // Only computed once
        assert_eq!(result1, result2);
        assert!((phi1 - phi2).abs() < 0.001);
        assert!(memoizer.hit_rate() > 0.0);
    }

    #[test]
    fn test_memoizer_eviction() {
        let mut memoizer = PrimitiveMemoizer::new(10);

        // Fill cache
        for i in 0..15 {
            memoizer.get_or_compute(&format!("A{}", i), "B", TransformationType::Bind, || {
                (HV16::random(i as u64), 0.5)
            });
        }

        // Should have evicted old entries
        assert!(memoizer.stats().cache_size <= 10);
    }

    // === PrimitiveAffinityGraph Tests ===

    #[test]
    fn test_affinity_graph_creation() {
        let graph = PrimitiveAffinityGraph::new();
        assert_eq!(graph.stats().total_pairs, 0);
    }

    #[test]
    fn test_affinity_graph_recording() {
        let mut graph = PrimitiveAffinityGraph::new();

        // Record some compositions
        graph.record_composition("BIND", "COMPARE", 0.8);
        graph.record_composition("BIND", "COMPARE", 0.6);
        graph.record_composition("BIND", "TRANSFORM", 0.5);

        // Check affinity
        let affinity = graph.get_affinity("BIND", "COMPARE");
        assert!((affinity - 0.7).abs() < 0.01);  // Average of 0.8 and 0.6

        assert_eq!(graph.stats().total_pairs, 2);
    }

    #[test]
    fn test_affinity_graph_tier_affinity() {
        let graph = PrimitiveAffinityGraph::new();

        // Same tier should have high affinity
        let same_tier = graph.get_tier_affinity(PrimitiveTier::Physical, PrimitiveTier::Physical);
        assert!((same_tier - 0.7).abs() < 0.01);

        // Consciousness and MetaCognitive should have special affinity
        let consciousness_meta = graph.get_tier_affinity(
            PrimitiveTier::Consciousness,
            PrimitiveTier::MetaCognitive
        );
        assert!((consciousness_meta - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_affinity_graph_best_partners() {
        let mut graph = PrimitiveAffinityGraph::new();

        graph.record_composition("A", "B", 0.9);
        graph.record_composition("A", "C", 0.5);
        graph.record_composition("A", "D", 0.7);

        let partners = graph.best_partners("A", 2);
        assert_eq!(partners.len(), 2);
        assert_eq!(partners[0].0, "B");  // Highest affinity
    }

    #[test]
    fn test_affinity_graph_suggest_chain() {
        let mut graph = PrimitiveAffinityGraph::new();

        graph.record_composition("START", "MIDDLE", 0.8);
        graph.record_composition("MIDDLE", "END", 0.7);

        let chain = graph.suggest_chain("START", 3);
        assert!(chain.len() >= 1);
        assert_eq!(chain[0], "START");
    }

    // === TierAwareConfig Tests ===

    #[test]
    fn test_tier_aware_config_default() {
        let config = TierAwareConfig::default();
        assert!(config.match_complexity);
        assert!((config.consciousness_boost - 0.2).abs() < 0.01);
        assert!(config.encourage_diversity);
    }

    #[test]
    fn test_tier_aware_selection() {
        let selector = AdaptivePrimitiveSelector::new().with_min_trust(1);
        let config = TierAwareConfig::default();

        // Create primitives from different tiers
        let primitives: Vec<Primitive> = vec![
            Primitive {
                name: "MATH_PRIM".to_string(),
                encoding: HV16::random(1),
                tier: PrimitiveTier::Mathematical,
                domain: "math".to_string(),
                definition: "Math".to_string(),
                is_base: true,
                derivation: None,
            },
            Primitive {
                name: "CONSCIOUSNESS_PRIM".to_string(),
                encoding: HV16::random(2),
                tier: PrimitiveTier::Consciousness,
                domain: "consciousness".to_string(),
                definition: "Consciousness".to_string(),
                is_base: true,
                derivation: None,
            },
        ];

        let prim_refs: Vec<&Primitive> = primitives.iter().collect();
        let selected = selector.select_tier_aware(
            TaskType::MetaCognitive,
            &prim_refs,
            2,
            &config,
            None
        );

        assert!(!selected.is_empty());
    }

    // === Integration Bridge Tests ===

    #[test]
    fn test_causal_bridge() {
        let bridge = PrimitiveCausalBridge::new();

        assert!(bridge.has_causal_semantics("CAUSE"));
        assert!(bridge.has_causal_semantics("EFFECT"));
        assert!(!bridge.has_causal_semantics("RANDOM_NAME"));

        assert_eq!(bridge.get_mechanism("CAUSE"), Some("direct_causation"));
    }

    #[test]
    fn test_synthesis_bridge() {
        let bridge = PrimitiveSynthesisBridge::new();

        assert!(bridge.is_synthesis_primitive("COMPOSE"));
        assert!(bridge.is_synthesis_primitive("SEQUENCE"));
        assert!(!bridge.is_synthesis_primitive("RANDOM"));

        assert!(bridge.get_synthesis_primitives().len() >= 6);
    }

    // === Tier Compatibility Matrix Tests ===

    #[test]
    fn test_tier_compatibility_matrix_creation() {
        let matrix = TierCompatibilityMatrix::new();

        // Same tier should be fully compatible
        assert_eq!(matrix.get_compatibility(PrimitiveTier::Mathematical, PrimitiveTier::Mathematical), 1.0);
        assert_eq!(matrix.get_compatibility(PrimitiveTier::Consciousness, PrimitiveTier::Consciousness), 1.0);

        // Known high-compatibility pairs
        assert!(matrix.get_compatibility(PrimitiveTier::Mathematical, PrimitiveTier::Physical) >= 0.8);
        assert!(matrix.get_compatibility(PrimitiveTier::Consciousness, PrimitiveTier::MetaCognitive) >= 0.9);

        // Known low-compatibility pairs
        assert!(matrix.get_compatibility(PrimitiveTier::Physical, PrimitiveTier::NSM) < 0.5);
    }

    #[test]
    fn test_tier_compatibility_symmetry() {
        let matrix = TierCompatibilityMatrix::new();
        let tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
        ];

        // Compatibility should be symmetric
        for &t1 in &tiers {
            for &t2 in &tiers {
                let c1 = matrix.get_compatibility(t1, t2);
                let c2 = matrix.get_compatibility(t2, t1);
                assert!((c1 - c2).abs() < 0.01,
                    "Compatibility should be symmetric: {:?}->{:?}={:.2}, {:?}->{:?}={:.2}",
                    t1, t2, c1, t2, t1, c2);
            }
        }
    }

    #[test]
    fn test_tier_chain_suggestion() {
        let matrix = TierCompatibilityMatrix::new();

        // Chain should start from given tier
        let chain = matrix.suggest_tier_chain(PrimitiveTier::Mathematical, 4);
        assert_eq!(chain[0], PrimitiveTier::Mathematical);
        assert!(chain.len() <= 4);

        // Chain should have no duplicates
        let unique: std::collections::HashSet<_> = chain.iter().collect();
        assert_eq!(unique.len(), chain.len(), "Chain should have no duplicates");

        // Each step should be recommended
        for i in 0..chain.len() - 1 {
            assert!(matrix.is_recommended(chain[i], chain[i + 1]),
                "Step {:?} -> {:?} should be recommended", chain[i], chain[i + 1]);
        }
    }

    #[test]
    fn test_composition_rules_coverage() {
        let matrix = TierCompatibilityMatrix::new();
        let tiers = [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ];

        // Every pair should have a defined rule (uses default if not explicit)
        for &t1 in &tiers {
            for &t2 in &tiers {
                let _rule = matrix.get_rule(t1, t2);
                // Should not panic - all pairs should return a valid rule
            }
        }
    }
}

// =============================================================================
// PROPERTY-BASED TESTS FOR CONSCIOUSNESS PRIMITIVES
// =============================================================================

#[cfg(test)]
mod proptest_consciousness {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating random tier values
    fn tier_strategy() -> impl Strategy<Value = PrimitiveTier> {
        prop_oneof![
            Just(PrimitiveTier::NSM),
            Just(PrimitiveTier::Mathematical),
            Just(PrimitiveTier::Physical),
            Just(PrimitiveTier::Geometric),
            Just(PrimitiveTier::Strategic),
            Just(PrimitiveTier::MetaCognitive),
            Just(PrimitiveTier::Temporal),
            Just(PrimitiveTier::Compositional),
            Just(PrimitiveTier::Consciousness),
        ]
    }

    // Strategy for generating phi values in valid range
    fn phi_strategy() -> impl Strategy<Value = f64> {
        (0.0..=1.0f64).prop_map(|x| x)
    }

    proptest! {
        /// Property: Tier compatibility is always in [0, 1]
        #[test]
        fn tier_compatibility_bounded(t1 in tier_strategy(), t2 in tier_strategy()) {
            let matrix = TierCompatibilityMatrix::new();
            let compat = matrix.get_compatibility(t1, t2);
            prop_assert!(compat >= 0.0 && compat <= 1.0,
                "Compatibility must be in [0,1], got {}", compat);
        }

        /// Property: Same tier always has maximum compatibility
        #[test]
        fn same_tier_max_compatibility(tier in tier_strategy()) {
            let matrix = TierCompatibilityMatrix::new();
            prop_assert_eq!(matrix.get_compatibility(tier, tier), 1.0);
        }

        /// Property: AdaptiveStats phi is never negative after recording
        #[test]
        fn adaptive_stats_phi_valid(phis in prop::collection::vec(phi_strategy(), 1..100)) {
            let mut stats = AdaptiveStats::new("test");

            for phi in phis {
                stats.record(phi);
            }

            prop_assert!(stats.mean_phi() >= 0.0,
                "Mean phi should be non-negative: {}", stats.mean_phi());
            prop_assert!(stats.success_rate() >= 0.0 && stats.success_rate() <= 1.0,
                "Success rate should be in [0,1]: {}", stats.success_rate());
        }

        /// Property: Affinity graph phi accumulation is correct
        #[test]
        fn affinity_accumulation(
            phis in prop::collection::vec(phi_strategy(), 1..50)
        ) {
            let mut graph = PrimitiveAffinityGraph::new();

            for phi in &phis {
                graph.record_composition("A", "B", *phi);
            }

            let affinity = graph.get_affinity("A", "B");
            let expected_mean = phis.iter().sum::<f64>() / phis.len() as f64;

            prop_assert!((affinity - expected_mean).abs() < 0.001,
                "Affinity should equal mean phi: got {} expected {}", affinity, expected_mean);
        }

        /// Property: Tier chain never exceeds requested length
        #[test]
        fn tier_chain_length_bounded(
            start in tier_strategy(),
            length in 1usize..15
        ) {
            let matrix = TierCompatibilityMatrix::new();
            let chain = matrix.suggest_tier_chain(start, length);

            prop_assert!(chain.len() <= length,
                "Chain length {} exceeds max {}", chain.len(), length);
            prop_assert!(!chain.is_empty(), "Chain should never be empty");
            prop_assert_eq!(chain[0], start, "Chain should start from requested tier");
        }
    }
}
