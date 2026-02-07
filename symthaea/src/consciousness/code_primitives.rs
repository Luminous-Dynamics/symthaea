//! Code Primitives Integration - Consciousness-Aware Code Reasoning
//!
//! This module bridges the Code tier primitives with the cognitive loop,
//! enabling Symthaea to reason about code through the same consciousness-aware
//! routing as all other cognitive tasks.
//!
//! # Architecture
//!
//! ```text
//! CodeIntent (user request)
//!     ↓
//! CodePrimitiveRouter
//!     ↓ select_primitives()
//! Vec<Primitive> [PARSE, ENCODE, GENERATE, ...]
//!     ↓ execute_with_phi()
//! CodeExecutionResult
//!     ↓
//! GeneratedCode (verified output)
//! ```
//!
//! # Integration with Cognitive Loop
//!
//! Code tasks flow through the standard primitive routing:
//! - `TaskType::Code` triggers code-specific primitive selection
//! - Phi is measured during code generation for quality control
//! - Meta-cognitive reflection activates on low Phi
//!
//! # Paradigm
//!
//! "HDC+CfC THINKS about code. Primitives provide the vocabulary.
//!  The cognitive loop orchestrates. LLM translates the result."

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::{HV16, Primitive, PrimitiveTier, ContinuousHV};

#[cfg(feature = "code_understanding")]
use crate::hdc::code_encoder::CodeHDEncoder;
#[cfg(feature = "code_understanding")]
use crate::language::code_intent::{CodeIntent, CodeIntentCategory};

use super::primitive_reasoning::{TaskType, TierAwareConfig, PrimitiveExecution, TransformationType};

/// Convert HV16 binary hypervector to ContinuousHV.
/// Maps each bit to -1.0 (0) or +1.0 (1) in bipolar encoding.
fn hv16_to_real_hv(hv: &HV16) -> ContinuousHV {
    let dim = hv.0.len() * 8;
    let mut values = Vec::with_capacity(dim);
    for byte in &hv.0 {
        for bit in (0..8).rev() {
            if (byte >> bit) & 1 == 1 {
                values.push(1.0);
            } else {
                values.push(-1.0);
            }
        }
    }
    ContinuousHV::from_vec(values)
}

/// Router that selects code primitives based on intent
#[derive(Debug, Clone)]
pub struct CodePrimitiveRouter {
    /// HDC dimension for encoding
    dim: usize,
    /// Cached primitive HVs for fast similarity lookup
    primitive_hvs: HashMap<String, ContinuousHV>,
    /// Configuration for primitive selection
    config: CodePrimitiveConfig,
}

/// Configuration for code primitive routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePrimitiveConfig {
    /// Minimum similarity threshold for primitive selection
    pub similarity_threshold: f32,
    /// Maximum primitives to select per intent
    pub max_primitives: usize,
    /// Whether to include cross-tier primitives (e.g., ABSTRACT from Mathematical)
    pub cross_tier_enabled: bool,
    /// Phi threshold below which to trigger meta-reflection
    pub phi_threshold: f32,
}

impl Default for CodePrimitiveConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.3,
            max_primitives: 5,
            cross_tier_enabled: true,
            phi_threshold: 0.4,
        }
    }
}

/// Result of executing code primitives
#[derive(Debug, Clone)]
pub struct CodeExecutionResult {
    /// Primitives that were executed
    pub primitives: Vec<PrimitiveExecution>,
    /// Resulting hypervector (code semantics)
    pub result_hv: ContinuousHV,
    /// Phi score of the execution (integration measure)
    pub phi: f32,
    /// Whether execution succeeded
    pub success: bool,
    /// Optional generated code
    pub generated_code: Option<String>,
    /// Diagnostic messages
    pub diagnostics: Vec<String>,
}

/// Maps code intents to primitive sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeOperation {
    /// Parse source code to AST
    Parse,
    /// Encode AST as hypervector
    Encode,
    /// Generate new code from spec
    Generate,
    /// Modify existing code
    Modify,
    /// Explain code semantics
    Explain,
    /// Find similar code patterns
    FindSimilar,
    /// Refactor code structure
    Refactor,
    /// Debug and diagnose issues
    Debug,
    /// Verify code correctness
    Verify,
}

impl CodeOperation {
    /// Get the primary primitives for this operation
    pub fn primary_primitives(&self) -> Vec<&'static str> {
        match self {
            CodeOperation::Parse => vec!["PARSE", "ENTITY", "ROLE"],
            CodeOperation::Encode => vec!["ENCODE", "BIND_SYMBOL", "ENTITY"],
            CodeOperation::Generate => vec!["GENERATE", "COMPOSE", "CODE_SEQUENCE"],
            CodeOperation::Modify => vec!["MUTATE", "VERIFY", "ENCODE"],
            CodeOperation::Explain => vec!["EXPLAIN", "TRACE", "INTENT"],
            CodeOperation::FindSimilar => vec!["CODE_SIMILARITY", "ENCODE", "ABSTRACT"],
            CodeOperation::Refactor => vec!["REFACTOR", "ABSTRACT", "SPECIALIZE"],
            CodeOperation::Debug => vec!["DEBUG", "TRACE", "BRANCH"],
            CodeOperation::Verify => vec!["VERIFY", "TYPE_CHECK", "ENCODE"],
        }
    }

    /// Get supporting primitives that may be composed
    pub fn supporting_primitives(&self) -> Vec<&'static str> {
        match self {
            CodeOperation::Parse => vec!["IMPORT", "BRANCH", "LOOP"],
            CodeOperation::Encode => vec!["ROLE", "CALL", "RETURN"],
            CodeOperation::Generate => vec!["SPECIALIZE", "BRANCH", "LOOP", "INTENT"],
            CodeOperation::Modify => vec!["REFACTOR", "CODE_SEQUENCE"],
            CodeOperation::Explain => vec!["ENTITY", "ROLE", "CALL"],
            CodeOperation::FindSimilar => vec!["ENTITY", "ROLE", "GENERALIZE"],
            CodeOperation::Refactor => vec!["VERIFY", "CODE_SEQUENCE", "COMPOSE"],
            CodeOperation::Debug => vec!["ENTITY", "CALL", "RETURN", "INTENT"],
            CodeOperation::Verify => vec!["PARSE", "CODE_SIMILARITY"],
        }
    }

    /// Get cross-tier primitives that enhance this operation
    pub fn cross_tier_primitives(&self) -> Vec<&'static str> {
        match self {
            CodeOperation::Parse => vec!["ATTEND", "STRUCTURE"],
            CodeOperation::Encode => vec!["PHENOMENAL_BINDING"],
            CodeOperation::Generate => vec!["INTEND", "SEQUENCE_OP", "FUNCTION"],
            CodeOperation::Modify => vec!["CAUSE", "EFFECT"],
            CodeOperation::Explain => vec!["QUALE", "MEANING"],
            CodeOperation::FindSimilar => vec!["ANALOGY", "PATTERN"],
            CodeOperation::Refactor => vec!["TRANSFORM", "EQUIVALENCE"],
            CodeOperation::Debug => vec!["CAUSE", "TRACE", "ERROR"],
            CodeOperation::Verify => vec!["PROOF", "CONSISTENCY"],
        }
    }
}

impl CodePrimitiveRouter {
    /// Create a new router with the given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            primitive_hvs: HashMap::new(),
            config: CodePrimitiveConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(dim: usize, config: CodePrimitiveConfig) -> Self {
        Self {
            dim,
            primitive_hvs: HashMap::new(),
            config,
        }
    }

    /// Cache primitive HVs from the global system for fast lookup
    pub fn cache_primitives(&mut self) {
        use symthaea_core::hdc::primitive_system::PrimitiveSystem;

        let system = PrimitiveSystem::global();

        // Cache all code tier primitives
        for primitive in system.get_tier(PrimitiveTier::Code) {
            let real_hv = hv16_to_real_hv(&primitive.encoding);
            self.primitive_hvs.insert(primitive.name.clone(), real_hv);
        }

        // Also cache commonly used cross-tier primitives
        let cross_tier = [
            "ATTEND", "INTEND", "CAUSE", "EFFECT", "FUNCTION",
            "SEQUENCE_OP", "ANALOGY", "PATTERN", "MEANING"
        ];

        for name in cross_tier {
            if let Some(primitive) = system.get(name) {
                let real_hv = hv16_to_real_hv(&primitive.encoding);
                self.primitive_hvs.insert(primitive.name.clone(), real_hv);
            }
        }
    }

    /// Select primitives for a code operation
    pub fn select_primitives(&self, operation: CodeOperation) -> Vec<Primitive> {
        use symthaea_core::hdc::primitive_system::PrimitiveSystem;

        let system = PrimitiveSystem::global();
        let mut primitives = Vec::new();

        // Get primary primitives
        for name in operation.primary_primitives() {
            if let Some(p) = system.get(name) {
                primitives.push(p.clone());
            }
        }

        // Add supporting primitives up to limit
        let remaining = self.config.max_primitives.saturating_sub(primitives.len());
        for name in operation.supporting_primitives().into_iter().take(remaining) {
            if let Some(p) = system.get(name) {
                primitives.push(p.clone());
            }
        }

        // Add cross-tier if enabled
        if self.config.cross_tier_enabled {
            let remaining = self.config.max_primitives.saturating_sub(primitives.len());
            for name in operation.cross_tier_primitives().into_iter().take(remaining) {
                if let Some(p) = system.get(name) {
                    primitives.push(p.clone());
                }
            }
        }

        primitives
    }

    /// Map a CodeIntent to a CodeOperation
    #[cfg(feature = "code_understanding")]
    pub fn intent_to_operation(intent: &CodeIntent) -> CodeOperation {
        match intent {
            CodeIntent::Create { .. } => CodeOperation::Generate,
            CodeIntent::Modify { .. } => CodeOperation::Modify,
            CodeIntent::Explain { .. } => CodeOperation::Explain,
            CodeIntent::Find { .. } => CodeOperation::FindSimilar,
            CodeIntent::Refactor { .. } => CodeOperation::Refactor,
            CodeIntent::Debug { .. } => CodeOperation::Debug,
        }
    }

    /// Map a CodeIntentCategory to a CodeOperation
    #[cfg(feature = "code_understanding")]
    pub fn category_to_operation(category: CodeIntentCategory) -> CodeOperation {
        match category {
            CodeIntentCategory::Create => CodeOperation::Generate,
            CodeIntentCategory::Modify => CodeOperation::Modify,
            CodeIntentCategory::Explain => CodeOperation::Explain,
            CodeIntentCategory::Find => CodeOperation::FindSimilar,
            CodeIntentCategory::Refactor => CodeOperation::Refactor,
            CodeIntentCategory::Debug => CodeOperation::Debug,
        }
    }

    /// Compose primitives into a unified representation
    pub fn compose_primitives(&self, primitives: &[Primitive]) -> ContinuousHV {
        if primitives.is_empty() {
            return ContinuousHV::random(self.dim, 0);
        }

        // Convert to ContinuousHV and bundle
        let hvs: Vec<ContinuousHV> = primitives
            .iter()
            .map(|p| hv16_to_real_hv(&p.encoding))
            .collect();

        // Bundle all primitives (requires references)
        let refs: Vec<&ContinuousHV> = hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Get the TaskType for code operations
    pub fn task_type() -> TaskType {
        TaskType::Code
    }

    /// Get tier-aware config optimized for code tasks
    pub fn code_optimized_config() -> TierAwareConfig {
        TierAwareConfig {
            code_weight: 2.0,          // Boost code tier
            compositional_weight: 1.5,  // Composition is important
            metacognitive_weight: 1.2,  // Self-reflection helps
            consciousness_weight: 1.0,
            temporal_weight: 1.0,
            nsm_weight: 0.8,           // Less emphasis on natural language
            math_weight: 1.0,
            physical_weight: 0.5,       // Physical reasoning less relevant
            geometric_weight: 0.5,
            strategic_weight: 0.8,
        }
    }
}

/// Execute code primitives with Phi measurement
pub struct CodePrimitiveExecutor {
    router: CodePrimitiveRouter,
    #[cfg(feature = "code_understanding")]
    _encoder: Option<CodeHDEncoder>,
}

impl CodePrimitiveExecutor {
    /// Create a new executor
    pub fn new(dim: usize) -> Self {
        let mut router = CodePrimitiveRouter::new(dim);
        router.cache_primitives();

        Self {
            router,
            #[cfg(feature = "code_understanding")]
            _encoder: Some(CodeHDEncoder::new(dim)),
        }
    }

    /// Execute an operation and measure Phi
    pub fn execute(&self, operation: CodeOperation) -> CodeExecutionResult {
        let primitives = self.router.select_primitives(operation);

        // Compose primitives
        let result_hv = self.router.compose_primitives(&primitives);

        // Create execution records
        let executions: Vec<PrimitiveExecution> = primitives
            .iter()
            .enumerate()
            .map(|(i, p)| PrimitiveExecution {
                primitive: p.clone(),
                input: p.encoding.clone(),
                output: primitives.first().map(|f| f.encoding.clone()).unwrap_or_else(|| p.encoding.clone()),
                transformation: TransformationType::Bundle,
                phi_contribution: 1.0 / (i + 1) as f64, // Decaying contribution
                timestamp: i as f64 * 0.1,
            })
            .collect();

        // Estimate Phi from composition coherence
        // High coherence = primitives work well together
        let phi = self.estimate_phi(&primitives);

        CodeExecutionResult {
            primitives: executions,
            result_hv,
            phi,
            success: true,
            generated_code: None,
            diagnostics: vec![
                format!("Selected {} primitives for {:?}", primitives.len(), operation),
                format!("Estimated Phi: {:.3}", phi),
            ],
        }
    }

    /// Estimate Phi from primitive composition
    fn estimate_phi(&self, primitives: &[Primitive]) -> f32 {
        if primitives.len() < 2 {
            return 0.5;
        }

        // Measure how well primitives integrate
        // Low similarity = independent = potentially high integration
        // Medium similarity = some structure = good integration
        // High similarity = redundant = low integration

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..primitives.len() {
            for j in (i + 1)..primitives.len() {
                let sim = primitives[i].encoding.similarity(&primitives[j].encoding);
                total_sim += sim;
                count += 1;
            }
        }

        if count == 0 {
            return 0.5;
        }

        let avg_sim = total_sim / count as f32;

        // Optimal integration is around 0.5 similarity
        // Too low = disconnected, too high = redundant
        let deviation = (avg_sim - 0.5).abs();
        let phi = 1.0 - deviation * 2.0;

        phi.clamp(0.0, 1.0)
    }

    /// Get primitives for an operation without executing
    pub fn get_primitives(&self, operation: CodeOperation) -> Vec<Primitive> {
        self.router.select_primitives(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_operation_primitives() {
        let op = CodeOperation::Generate;
        let primary = op.primary_primitives();

        assert!(primary.contains(&"GENERATE"));
        assert!(primary.contains(&"COMPOSE"));
    }

    #[test]
    fn test_router_creation() {
        let router = CodePrimitiveRouter::new(512);
        assert_eq!(router.dim, 512);
    }

    #[test]
    fn test_select_primitives() {
        let mut router = CodePrimitiveRouter::new(512);
        router.cache_primitives();

        let primitives = router.select_primitives(CodeOperation::Parse);
        assert!(!primitives.is_empty());

        // Should include PARSE
        assert!(primitives.iter().any(|p| p.name == "PARSE"));
    }

    #[test]
    fn test_compose_primitives() {
        let mut router = CodePrimitiveRouter::new(512);
        router.cache_primitives();

        let primitives = router.select_primitives(CodeOperation::Generate);
        let composed = router.compose_primitives(&primitives);

        // Composed result should have valid dimension (HV16 is always 16384-bit)
        // HV16::BYTES = 2048, so dimension = 2048 * 8 = 16384
        assert_eq!(composed.dim(), 16384);
        let norm: f32 = composed.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0, "Composed HV should have non-zero norm");
    }

    #[test]
    fn test_executor() {
        let executor = CodePrimitiveExecutor::new(512);
        let result = executor.execute(CodeOperation::Debug);

        assert!(result.success);
        assert!(!result.primitives.is_empty());
        assert!(result.phi > 0.0);
    }

    #[test]
    fn test_code_optimized_config() {
        let config = CodePrimitiveRouter::code_optimized_config();

        // Code weight should be highest
        assert!(config.code_weight > config.nsm_weight);
        assert!(config.code_weight > config.physical_weight);
    }

    #[test]
    fn test_all_operations_have_primitives() {
        let operations = [
            CodeOperation::Parse,
            CodeOperation::Encode,
            CodeOperation::Generate,
            CodeOperation::Modify,
            CodeOperation::Explain,
            CodeOperation::FindSimilar,
            CodeOperation::Refactor,
            CodeOperation::Debug,
            CodeOperation::Verify,
        ];

        for op in operations {
            assert!(!op.primary_primitives().is_empty(), "{:?} has no primitives", op);
        }
    }
}
