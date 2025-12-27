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
    fn measure_phi_contribution(&self, input: &HV16, output: &HV16) -> Result<f64> {
        // Φ measures information integration
        // For a transformation step, we measure how much the transformation
        // integrated information from input to output

        let mut phi_computer = IntegratedInformation::new();

        // Create component set: [input, output]
        let components = vec![input.clone(), output.clone()];

        // Compute Φ for this transformation
        let phi = phi_computer.compute_phi(&components);

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
}
