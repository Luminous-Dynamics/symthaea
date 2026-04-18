// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language module - natural language processing and generation
//!
//! Core language capabilities including:
//! - Emotional analysis and response generation
//! - LLM orchestration
//! - Domain-agnostic plugin system
//! - NixOS configuration parsing (via plugin)
//! - Consciousness monitoring

// Core working modules
pub mod consciousness_prompts;
pub mod domain_plugin;
pub mod emotional_core;
pub mod llm_backend;
pub mod llm_organ;
pub mod nix_parser;
pub mod phi_monitor;
pub mod semantic_intent;

// Domain plugins
pub mod code_assistant_plugin;
pub mod general_assistant_plugin;
pub mod math_plugin;
pub mod nixos_plugin;
pub mod programming_plugin;
pub mod research_plugin;

// Narrative prompt compiler (CfC Ghost Signal → LLM writing instructions)
pub mod narrative_compiler;

// LLM provider backends
pub mod anthropic_backend;
#[cfg(feature = "full_language")]
pub mod candle_backend;
#[cfg(feature = "ssm_language")]
pub mod distillation;
pub mod openai_backend;
#[cfg(feature = "ssm_language")]
pub mod ssm_backend;

// Intelligent code generation dispatch (consciousness-routed backend selection)
pub mod intelligent_dispatcher;

// Machine-verifiable code certificates — the competitive moat
#[cfg(feature = "code_generation")]
pub mod code_certificate;

// Unified code orchestrator — routes requests through native backends before LLM fallback
#[cfg(feature = "code_generation")]
pub mod code_orchestrator;

// Pattern library bootstrap — parse Rust source files to populate HDC pattern library
#[cfg(feature = "code_generation")]
pub mod pattern_bootstrap;

// Code understanding & generation (Phase: Consciousness-Aware Code)
#[cfg(feature = "code_generation")]
pub mod algorithm_encoder;
#[cfg(feature = "code_generation")]
pub mod algorithm_training;
#[cfg(feature = "code_generation")]
pub mod analogy_generation;
#[cfg(feature = "code_generation")]
pub mod code_discovery;
#[cfg(feature = "code_generation")]
pub mod code_domain_plugin;
#[cfg(feature = "code_generation")]
pub mod code_executor;
#[cfg(feature = "code_generation")]
pub mod code_generator;
#[cfg(feature = "code_generation")]
pub mod code_intent;
#[cfg(feature = "code_generation")]
pub mod code_parser;
#[cfg(feature = "code_generation")]
pub mod code_verifier;
#[cfg(feature = "code_generation")]
pub mod emitters;
#[cfg(feature = "code_generation")]
pub mod epistemic_generation;
#[cfg(feature = "code_generation")]
pub mod hv_code_decoder;
#[cfg(feature = "code_generation")]
pub mod learned_idioms;
#[cfg(feature = "code_generation")]
pub mod nix_code_parser;
#[cfg(feature = "code_generation")]
pub mod nix_codegen;
#[cfg(feature = "code_generation")]
pub mod nix_eval_corpus;
#[cfg(feature = "code_generation")]
pub mod nix_kg;
#[cfg(feature = "code_generation")]
pub mod nix_scorer;
#[cfg(feature = "code_generation")]
pub mod nixos_search;
#[cfg(feature = "code_generation")]
pub mod nixpkgs_index;
#[cfg(feature = "code_generation")]
pub mod parser_registry;
pub mod program_node_translator;
#[cfg(feature = "code_generation")]
pub mod python_parser;
#[cfg(feature = "code_generation")]
pub mod rust_parser;
#[cfg(feature = "code_generation")]
pub mod sequencer_benchmark;
#[cfg(feature = "code_generation")]
pub mod sequencer_training;
#[cfg(feature = "code_generation")]
pub mod triune_intent;
#[cfg(feature = "code_generation")]
pub mod type_causal_model;
#[cfg(feature = "code_generation")]
pub mod verified_generation;

// Modules needing HDC submodules that don't exist yet (cfg-gated)
#[cfg(feature = "full_language")]
pub mod learning_persistence;
#[cfg(feature = "full_language")]
pub mod meta_conscious_llm_bridge;

// Enhanced modules (need API alignment - cfg-gated)
// REMOVED: enhanced_consciousness (~1,936 LOC) — zero external usage, dead code
// REMOVED: semantic_enrichment (~2,391 LOC) — zero external usage, dead code
// Files retained on disk for reference; just not compiled.
#[cfg(feature = "full_language")]
pub mod multi_theory_consciousness;

// Re-exports
pub use domain_plugin::ErrorDiagnosis as DomainErrorDiagnosis;
pub use domain_plugin::{
    DomainPlugin, DomainPrompts, Entity, ErrorLocation, GenericPlugin, IntentPrototypes,
    PluginRegistry, RiskLevel, ValidationResult,
};
pub use emotional_core::{
    EmotionalAnalysis, EmotionalCore, EmotionalCoreConfig, EmotionalResponse,
};
pub use llm_organ::{
    ConversationMessage, LLMGenerationResult, LLMOrgan, LLMOrganConfig, LLMQuery, MessageRole,
    QueryType, TRANSLATION_SYSTEM_PROMPT,
};
pub use math_plugin::MathPlugin;
pub use nix_parser::{NixConfig, NixOption, NixParser, NixValue};
pub use nixos_plugin::NixOsPlugin;
pub use programming_plugin::ProgrammingPlugin;
pub use semantic_intent::{IntentCategory, IntentClassification, SemanticIntentClassifier};
// Export backend module for creating custom backends
pub use anthropic_backend::AnthropicBackend;
pub use code_assistant_plugin::CodeAssistantPlugin;
pub use general_assistant_plugin::GeneralAssistantPlugin;
pub use llm_backend::{
    create_backend_from_env, default_backend, simulated_backend, GenerationParams, LLMBackend,
    OllamaBackend, SimulatedBackend,
};
pub use narrative_compiler::{
    generate_narrative, NarrativeCompiler, NarrativeOutput, NarrativeThought,
    NARRATIVE_SYSTEM_PROMPT,
};
pub use openai_backend::OpenAiBackend;
pub use research_plugin::ResearchPlugin;

// ============================================================================
// NixOS Error Diagnoser (for shell module integration)
// ============================================================================

/// Categories of NixOS errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixErrorCategory {
    /// Evaluation errors (undefined variables, type errors)
    Evaluation,
    /// Build failures
    Build,
    /// Conflicting options or packages
    Conflict,
    /// Missing resources (packages, files)
    MissingResource,
    /// Permission errors
    Permission,
    /// Flake-specific errors
    Flake,
    /// Unknown error
    Unknown,
}

impl NixErrorCategory {
    /// Get the category name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Self::Evaluation => "Evaluation Error",
            Self::Build => "Build Error",
            Self::Conflict => "Conflict Error",
            Self::MissingResource => "Missing Resource",
            Self::Permission => "Permission Error",
            Self::Flake => "Flake Error",
            Self::Unknown => "Unknown Error",
        }
    }
}

/// Specific types of NixOS errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NixErrorType {
    /// Missing semicolon
    MissingSemicolon,
    /// Undefined variable
    UndefinedVariable(String),
    /// Type mismatch
    TypeMismatch { expected: String, found: String },
    /// Missing attribute
    MissingAttribute(String),
    /// Package not found
    PackageNotFound(String),
    /// Build failed
    BuildFailed(String),
    /// Infinite recursion
    InfiniteRecursion,
    /// Hash mismatch
    HashMismatch,
    /// Network error
    NetworkError(String),
    /// Other error
    Other(String),
}

impl NixErrorType {
    /// Get the error type name as a string
    pub fn name(&self) -> &str {
        match self {
            Self::MissingSemicolon => "Missing Semicolon",
            Self::UndefinedVariable(_v) => "Undefined Variable",
            Self::TypeMismatch { .. } => "Type Mismatch",
            Self::MissingAttribute(_) => "Missing Attribute",
            Self::PackageNotFound(_) => "Package Not Found",
            Self::BuildFailed(_) => "Build Failed",
            Self::InfiniteRecursion => "Infinite Recursion",
            Self::HashMismatch => "Hash Mismatch",
            Self::NetworkError(_) => "Network Error",
            Self::Other(_) => "Error",
        }
    }
}

/// Risk level for suggested fixes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum FixRiskLevel {
    /// Safe to apply automatically
    #[default]
    Safe,
    /// Low risk, review recommended
    Low,
    /// Medium risk, careful review needed
    Medium,
    /// High risk, manual intervention recommended
    High,
    /// Critical risk, expert review required
    Critical,
}

/// A suggested fix for an error
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// Description of what this fix does
    pub description: String,
    /// Command to run (if applicable)
    pub command: Option<String>,
    /// Risk level of applying this fix
    pub risk: FixRiskLevel,
    /// Confidence that this fix will work (0.0-1.0)
    pub confidence: f32,
    /// Whether this is the primary/recommended fix
    pub primary: bool,
}

/// Diagnosis result from error analysis
#[derive(Debug, Clone)]
pub struct ErrorDiagnosis {
    /// Error category
    pub category: NixErrorCategory,
    /// Specific error type
    pub error_type: NixErrorType,
    /// Human-readable explanation
    pub explanation: String,
    /// Likely causes (ordered by probability)
    pub likely_causes: Vec<String>,
    /// Suggested fixes
    pub fixes: Vec<SuggestedFix>,
    /// File location if known
    pub location: Option<String>,
    /// Affected configuration paths
    pub affected_configs: Vec<String>,
    /// Confidence in diagnosis (0.0-1.0)
    pub confidence: f32,
}

/// NixOS error diagnoser
#[derive(Debug)]
pub struct NixErrorDiagnoser {
    /// Known error patterns
    patterns: Vec<ErrorPattern>,
}

#[derive(Debug)]
struct ErrorPattern {
    regex: regex::Regex,
    category: NixErrorCategory,
    explanation_template: String,
}

impl NixErrorDiagnoser {
    /// Create a new error diagnoser
    pub fn new() -> Self {
        let patterns = vec![
            ErrorPattern {
                regex: regex::Regex::new(r"error: syntax error, unexpected")
                    .expect("valid regex literal"),
                category: NixErrorCategory::Evaluation,
                explanation_template: "Syntax error in Nix expression".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"undefined variable '(\w+)'")
                    .expect("valid regex literal"),
                category: NixErrorCategory::Evaluation,
                explanation_template: "Variable is not defined in scope".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"attribute '(\w+)' missing")
                    .expect("valid regex literal"),
                category: NixErrorCategory::Evaluation,
                explanation_template: "Required attribute is missing".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"error: hash mismatch").expect("valid regex literal"),
                category: NixErrorCategory::Build,
                explanation_template: "Downloaded file hash doesn't match expected".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"builder.*failed").expect("valid regex literal"),
                category: NixErrorCategory::Build,
                explanation_template: "Package build process failed".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"permission denied").expect("valid regex literal"),
                category: NixErrorCategory::Permission,
                explanation_template: "Insufficient permissions for this operation".to_string(),
            },
            ErrorPattern {
                regex: regex::Regex::new(r"flake.*error|error.*flake")
                    .expect("valid regex literal"),
                category: NixErrorCategory::Flake,
                explanation_template: "Flake configuration error".to_string(),
            },
        ];

        Self { patterns }
    }

    /// Diagnose an error from output text
    pub fn diagnose(&self, error_text: &str) -> ErrorDiagnosis {
        for pattern in &self.patterns {
            if pattern.regex.is_match(error_text) {
                return ErrorDiagnosis {
                    category: pattern.category,
                    error_type: NixErrorType::Other(
                        error_text.lines().next().unwrap_or("").to_string(),
                    ),
                    explanation: pattern.explanation_template.clone(),
                    likely_causes: vec!["See error message for details".to_string()],
                    fixes: vec![],
                    location: self.extract_location(error_text),
                    affected_configs: vec![],
                    confidence: 0.70,
                };
            }
        }

        // Default unknown error
        ErrorDiagnosis {
            category: NixErrorCategory::Unknown,
            error_type: NixErrorType::Other(
                error_text
                    .lines()
                    .next()
                    .unwrap_or("Unknown error")
                    .to_string(),
            ),
            explanation: "Could not automatically diagnose this error".to_string(),
            likely_causes: vec!["Error pattern not recognized".to_string()],
            fixes: vec![],
            location: self.extract_location(error_text),
            affected_configs: vec![],
            confidence: 0.20,
        }
    }

    /// Extract file location from error text
    fn extract_location(&self, error_text: &str) -> Option<String> {
        // Look for patterns like "at /path/to/file.nix:123:45"
        let location_re = regex::Regex::new(r"at\s+(/[^:]+):(\d+):(\d+)").ok()?;
        if let Some(caps) = location_re.captures(error_text) {
            return Some(format!("{}:{}:{}", &caps[1], &caps[2], &caps[3]));
        }
        None
    }
}

impl Default for NixErrorDiagnoser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONSCIOUSNESS LANGUAGE TYPES (for integration_module compatibility)
// ============================================================================
// The integration module expects these types for conscious language processing.
// These are stub implementations to enable module compilation.

use symthaea_core::hdc::ContinuousHV;

/// Consciousness state level for language processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsciousnessStateLevel {
    /// Minimal awareness
    #[default]
    Dormant,
    /// Basic reactive awareness
    Reactive,
    /// Active processing awareness
    Active,
    /// Deep reflective awareness
    Reflective,
    /// Full metacognitive awareness
    Metacognitive,
}

/// Consciousness quadrant for integrated processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsciousnessQuadrant {
    /// Analytical/logical processing
    #[default]
    Analytical,
    /// Intuitive/pattern processing
    Intuitive,
    /// Emotional/affective processing
    Emotional,
    /// Somatic/embodied processing
    Somatic,
}

/// Execution strategy type for NixOS commands (simple enum for basic categorization)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionStrategyType {
    /// Direct execution
    #[default]
    Direct,
    /// Staged/phased execution
    Staged,
    /// Dry-run first
    DryRun,
    /// Interactive confirmation
    Interactive,
    /// Batch execution
    Batch,
    /// Confident quadrant
    Confident,
    /// Autopilot quadrant
    Autopilot,
    /// Curious quadrant
    Curious,
    /// Lost quadrant
    Lost,
}

/// Execution strategy based on 2D consciousness quadrants (Φ × Confidence)
///
/// - Confident: High Φ + High Confidence → execute with full trust
/// - Autopilot: Low Φ + High Confidence → pattern-matched routine, execute efficiently
/// - Curious: High Φ + Low Confidence → exploring, dry-run first
/// - Lost: Low Φ + Low Confidence → genuinely confused, request help
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// High Φ + High Confidence: Deep understanding, execute with full trust
    Confident {
        /// Execute the action immediately
        execute_immediately: bool,
        /// Validate the result after execution
        validate_after: bool,
        /// Provide reasoning with the response
        explain_reasoning: bool,
    },
    /// Low Φ + High Confidence: Pattern-matched routine, execute efficiently
    Autopilot {
        /// Execute the action efficiently
        execute_efficiently: bool,
        /// Monitor for errors during execution
        monitor_for_errors: bool,
        /// Keep response minimal
        minimal_response: bool,
    },
    /// High Φ + Low Confidence: Deep processing but uncertain, explore first
    Curious {
        /// Do a dry-run before executing
        explore_first: bool,
        /// Targeted questions to clarify
        targeted_questions: Vec<ClarifyingQuestion>,
        /// Share the reasoning process
        share_reasoning: bool,
    },
    /// Low Φ + Low Confidence: Genuinely confused, don't execute
    Lost {
        /// Request help from user
        request_help: bool,
        /// Generic questions to gather context
        generic_questions: Vec<ClarifyingQuestion>,
        /// Admit confusion explicitly
        admit_confusion: bool,
    },
}

impl Default for ExecutionStrategy {
    fn default() -> Self {
        Self::Curious {
            explore_first: true,
            targeted_questions: Vec::new(),
            share_reasoning: true,
        }
    }
}

impl ExecutionStrategy {
    /// Create a safe default strategy (Curious with explore_first)
    pub fn safe_default() -> Self {
        Self::Curious {
            explore_first: true,
            targeted_questions: Vec::new(),
            share_reasoning: true,
        }
    }

    /// Create a confident strategy
    pub fn confident() -> Self {
        Self::Confident {
            execute_immediately: true,
            validate_after: true,
            explain_reasoning: true,
        }
    }

    /// Create an autopilot strategy
    pub fn autopilot() -> Self {
        Self::Autopilot {
            execute_efficiently: true,
            monitor_for_errors: true,
            minimal_response: true,
        }
    }

    /// Create a lost strategy
    pub fn lost() -> Self {
        Self::Lost {
            request_help: true,
            generic_questions: Vec::new(),
            admit_confusion: true,
        }
    }
}

impl From<&ExecutionStrategy> for ExecutionStrategyType {
    fn from(strategy: &ExecutionStrategy) -> Self {
        match strategy {
            ExecutionStrategy::Confident { .. } => ExecutionStrategyType::Confident,
            ExecutionStrategy::Autopilot { .. } => ExecutionStrategyType::Autopilot,
            ExecutionStrategy::Curious { .. } => ExecutionStrategyType::Curious,
            ExecutionStrategy::Lost { .. } => ExecutionStrategyType::Lost,
        }
    }
}

/// A clarifying question to gather more information
#[derive(Debug, Clone)]
pub struct ClarifyingQuestion {
    /// The question text
    pub question: String,
    /// Why this question matters
    pub rationale: String,
    /// Possible answer options (if applicable)
    pub options: Vec<String>,
    /// Priority of this question
    pub priority: u8,
    /// Default answer if user skips
    pub default: Option<String>,
}

/// NixOS intent as an enum (for integration module compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NixOSIntent {
    /// Install packages
    Install,
    /// Remove packages
    Remove,
    /// Search for packages
    Search,
    /// Upgrade system or packages
    Upgrade,
    /// Configure system options
    Configure,
    /// Garbage collect old generations
    GarbageCollect,
    /// Flake operations (init, update, etc.)
    FlakeOp,
    /// Rollback to previous generation
    Rollback,
    /// List installed packages or generations
    List,
    /// Get info about a package or option
    Info,
    /// Build a derivation
    Build,
    /// Switch to a new configuration
    Switch,
    /// Unknown intent
    #[default]
    Unknown,
}

impl NixOSIntent {
    /// Whether this intent is a read-only query
    pub fn is_query(&self) -> bool {
        matches!(self, Self::Search | Self::List | Self::Info)
    }

    /// Get the action name as a string
    pub fn action_name(&self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::Remove => "remove",
            Self::Search => "search",
            Self::Upgrade => "upgrade",
            Self::Configure => "configure",
            Self::GarbageCollect => "gc",
            Self::FlakeOp => "flake",
            Self::Rollback => "rollback",
            Self::List => "list",
            Self::Info => "info",
            Self::Build => "build",
            Self::Switch => "switch",
            Self::Unknown => "unknown",
        }
    }
}

/// Understanding of NixOS-specific intent (struct form for backwards compatibility)
#[derive(Debug, Clone, Default)]
pub struct NixOSIntentStruct {
    /// Primary action (install, remove, configure, etc.)
    pub action: String,
    /// Target packages or options
    pub targets: Vec<String>,
    /// Configuration scope (user, system, flake)
    pub scope: String,
    /// Whether this is a query or mutation
    pub is_query: bool,
    /// Confidence in intent recognition
    pub confidence: f32,
}

/// Understanding result for NixOS queries
#[derive(Debug, Clone, Default)]
pub struct NixOSUnderstanding {
    /// Recognized intent (enum form)
    pub intent: NixOSIntent,
    /// Extracted entities (packages, options, paths)
    pub entities: Vec<String>,
    /// Relevant context
    pub context: String,
    /// Any ambiguities detected
    pub ambiguities: Vec<String>,
    /// Suggested clarifications
    pub clarifications: Vec<ClarifyingQuestion>,
    /// Parameters extracted from input (key-value pairs)
    pub parameters: std::collections::HashMap<String, String>,
    /// Human-readable description of the intent
    pub description: String,
    /// Confidence in understanding (0.0-1.0)
    pub confidence: f32,
}

/// Feedback from action execution (expanded for integration module)
#[derive(Debug, Clone)]
pub struct ActionOutcomeFeedback {
    /// Original user input that led to this action
    pub original_input: String,
    /// Which quadrant the decision was made in
    pub decided_in_quadrant: ConsciousnessQuadrant,
    /// Strategy type that was used
    pub strategy_used: ExecutionStrategyType,
    /// Φ level at decision time
    pub phi_at_decision: f64,
    /// Confidence at decision time
    pub confidence_at_decision: f64,
    /// Whether the action succeeded
    pub action_succeeded: bool,
    /// Φ level after execution
    pub phi_after: f64,
    /// Error message if action failed
    pub error_message: Option<String>,
    /// Whether this was a dry-run
    pub was_dry_run: bool,
    /// User feedback (positive/negative) if provided
    pub user_feedback: Option<bool>,
    // Legacy fields for backwards compatibility
    /// Whether the action succeeded (legacy alias)
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// User satisfaction (if provided)
    pub user_satisfaction: Option<f32>,
    /// Any errors encountered
    pub errors: Vec<String>,
    /// Lessons learned
    pub lessons: Vec<String>,
}

impl Default for ActionOutcomeFeedback {
    fn default() -> Self {
        Self {
            original_input: String::new(),
            decided_in_quadrant: ConsciousnessQuadrant::Analytical,
            strategy_used: ExecutionStrategyType::Direct,
            phi_at_decision: 0.5,
            confidence_at_decision: 0.5,
            action_succeeded: true,
            phi_after: 0.5,
            error_message: None,
            was_dry_run: false,
            user_feedback: None,
            success: true,
            execution_time_ms: 0,
            user_satisfaction: None,
            errors: Vec::new(),
            lessons: Vec::new(),
        }
    }
}

/// Result from conscious understanding process
#[derive(Debug, Clone)]
pub struct ConsciousUnderstandingResult {
    /// NixOS-specific understanding (legacy field name)
    pub nixos: NixOSUnderstanding,
    /// NixOS-specific understanding (alias for integration module)
    pub nix_understanding: NixOSUnderstanding,
    /// Emotional context
    pub emotional_context: Option<EmotionalAnalysis>,
    /// Consciousness state during processing
    pub consciousness_state: ConsciousnessStateLevel,
    /// Active quadrants
    pub active_quadrants: Vec<ConsciousnessQuadrant>,
    /// Current consciousness quadrant (2D space)
    pub quadrant: ConsciousnessQuadrant,
    /// Embedding representation
    pub embedding: Option<ContinuousHV>,
    /// Processing confidence
    pub confidence: f32,
    /// Epistemic confidence (certainty about interpretation)
    pub epistemic_confidence: f64,
    /// Consciousness Φ (integration depth)
    pub consciousness_phi: f64,
    /// Unified free energy from active inference
    pub unified_free_energy: f64,
    /// Recommended execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Recommended strategy (legacy alias)
    pub recommended_strategy: ExecutionStrategy,
    /// Clarifying questions if confidence is low
    pub clarifying_questions: Vec<ClarifyingQuestion>,
    /// HDC-based semantic intent classification (if available)
    pub intent_classification: Option<IntentClassification>,

    /// Primitive tiers likely active for this input (grounding step).
    /// Maps the semantic intent to ontological primitive tiers, providing
    /// a bridge between language understanding and primitive-based reasoning.
    pub primitive_tiers: Vec<String>,

    /// Concrete executable primitives (the "Hands").
    pub primitives: Vec<String>,
}

impl Default for ConsciousUnderstandingResult {
    fn default() -> Self {
        let nix_understanding = NixOSUnderstanding::default();
        let strategy = ExecutionStrategy::safe_default();
        Self {
            nixos: nix_understanding.clone(),
            nix_understanding,
            emotional_context: None,
            consciousness_state: ConsciousnessStateLevel::Active,
            active_quadrants: vec![ConsciousnessQuadrant::Analytical],
            quadrant: ConsciousnessQuadrant::Analytical,
            embedding: None,
            confidence: 0.5,
            epistemic_confidence: 0.5,
            consciousness_phi: 0.5,
            unified_free_energy: 1.0,
            execution_strategy: strategy.clone(),
            recommended_strategy: strategy,
            clarifying_questions: Vec::new(),
            intent_classification: None,
            primitive_tiers: Vec::new(),
            primitives: Vec::new(),
        }
    }
}

/// Configuration for consciousness language core
#[derive(Debug, Clone)]
pub struct ConsciousnessLanguageConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Enable emotional processing
    pub emotional_enabled: bool,
    /// Enable metacognitive reflection
    pub metacognition_enabled: bool,
    /// LLM configuration
    pub llm_config: LLMOrganConfig,
    /// Minimum confidence threshold
    pub confidence_threshold: f32,
}

impl Default for ConsciousnessLanguageConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            emotional_enabled: true,
            metacognition_enabled: true,
            llm_config: LLMOrganConfig::default(),
            confidence_threshold: 0.6,
        }
    }
}

/// Core consciousness language processor
#[allow(dead_code)] // Fields reserved for language processing
pub struct ConsciousnessLanguageCore {
    /// Configuration
    config: ConsciousnessLanguageConfig,
    /// LLM organ for generation
    llm: LLMOrgan,
    /// Emotional processor
    emotional: EmotionalCore,
    /// Current consciousness state
    state: ConsciousnessStateLevel,
    /// Error diagnoser
    diagnoser: NixErrorDiagnoser,
    /// HDC-based semantic intent classifier
    intent_classifier: SemanticIntentClassifier,
    /// Current Φ level
    current_phi: f64,
    /// Current epistemic confidence
    current_confidence: f64,
    /// Last input processed (for feedback)
    last_input: String,
    /// Last quadrant (for feedback)
    last_quadrant: ConsciousnessQuadrant,
}

impl std::fmt::Debug for ConsciousnessLanguageCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsciousnessLanguageCore")
            .field("state", &self.state)
            .field("current_phi", &self.current_phi)
            .field("current_confidence", &self.current_confidence)
            .finish_non_exhaustive()
    }
}

impl ConsciousnessLanguageCore {
    /// Create a new consciousness language core
    pub fn new(config: ConsciousnessLanguageConfig) -> Self {
        Self {
            llm: LLMOrgan::new(config.llm_config.clone()),
            emotional: EmotionalCore::new(EmotionalCoreConfig::default()),
            state: ConsciousnessStateLevel::Active,
            diagnoser: NixErrorDiagnoser::new(),
            intent_classifier: SemanticIntentClassifier::new(),
            current_phi: 0.5,
            current_confidence: 0.5,
            last_input: String::new(),
            last_quadrant: ConsciousnessQuadrant::Analytical,
            config,
        }
    }

    /// Create with config (alias for new, for API compatibility)
    pub fn with_config(config: ConsciousnessLanguageConfig) -> Self {
        Self::new(config)
    }

    /// Process input (alias for understand, for API compatibility)
    pub fn process(&mut self, input: &str) -> ConsciousUnderstandingResult {
        self.understand(input)
    }

    /// Process input with consciousness awareness
    pub fn understand(&mut self, input: &str) -> ConsciousUnderstandingResult {
        // Save input for feedback
        self.last_input = input.to_string();

        // HDC-based semantic intent classification (runs BEFORE keyword extraction)
        let intent_classification = self.intent_classifier.classify(input);

        // Basic understanding flow
        let emotional_context = if self.config.emotional_enabled {
            Some(self.emotional.analyze(input))
        } else {
            None
        };

        // Use semantic classification to guide keyword-based intent extraction:
        // If the classifier says this is NOT NixOS with confidence > 0.5,
        // and the keyword extractor also returns Unknown, keep it as Unknown
        // (routes to general LLM). Otherwise, proceed with keyword extraction.
        let intent = if intent_classification.category != IntentCategory::NixOS
            && intent_classification.confidence > 0.5
        {
            // Non-NixOS classification with decent confidence:
            // still run keyword extraction, but if it returns Unknown, trust the classifier
            let keyword_intent = self.extract_nixos_intent(input);
            if keyword_intent == NixOSIntent::Unknown {
                NixOSIntent::Unknown
            } else {
                // Keyword extraction found something specific - trust it
                keyword_intent
            }
        } else {
            self.extract_nixos_intent(input)
        };

        let entities = self.extract_entities(input);

        // Build parameters map
        let mut parameters = std::collections::HashMap::new();
        if let Some(first_entity) = entities.first() {
            match intent {
                NixOSIntent::Install | NixOSIntent::Remove => {
                    parameters.insert("package".to_string(), first_entity.clone());
                }
                NixOSIntent::Search => {
                    parameters.insert("query".to_string(), first_entity.clone());
                }
                _ => {}
            }
        }

        // Determine confidence based on intent clarity
        let confidence = match intent {
            NixOSIntent::Unknown => 0.3,
            _ => 0.7,
        };

        // Generate description
        let description = match intent {
            NixOSIntent::Install => format!("Install package: {entities:?}"),
            NixOSIntent::Remove => format!("Remove package: {entities:?}"),
            NixOSIntent::Search => format!("Search for: {entities:?}"),
            NixOSIntent::Upgrade => "Upgrade system".to_string(),
            NixOSIntent::Configure => "Configure system".to_string(),
            NixOSIntent::List => "List packages".to_string(),
            _ => format!("Process: {input}"),
        };

        let nix_understanding = NixOSUnderstanding {
            intent,
            entities,
            context: input.to_string(),
            ambiguities: Vec::new(),
            clarifications: Vec::new(),
            parameters,
            description,
            confidence,
        };

        // Determine consciousness quadrant based on Φ and confidence
        let (quadrant, strategy) =
            self.determine_quadrant_and_strategy(self.current_phi, confidence as f64);

        self.last_quadrant = quadrant;
        self.current_confidence = confidence as f64;

        // Generate clarifying questions if uncertain
        let clarifying_questions = if confidence < self.config.confidence_threshold {
            vec![ClarifyingQuestion {
                question: format!("Could you clarify what you'd like to do with '{input}'?"),
                rationale: "The intent wasn't clear from the input".to_string(),
                options: vec![
                    "Install".to_string(),
                    "Remove".to_string(),
                    "Search".to_string(),
                ],
                priority: 1,
                default: None,
            }]
        } else {
            Vec::new()
        };

        // Primitive grounding: identify which ontological primitive tiers
        // are likely active for this input based on the semantic classification.
        // This bridges language understanding with the 9-tier primitive system.
        let primitive_tiers = Self::ground_to_primitive_tiers(&intent_classification);

        // Concrete executable primitives: determine what actions to take
        let primitives = Self::ground_to_primitives(input, &intent_classification);

        ConsciousUnderstandingResult {
            nixos: nix_understanding.clone(),
            nix_understanding,
            emotional_context,
            consciousness_state: self.state,
            active_quadrants: vec![quadrant],
            quadrant,
            embedding: None,
            confidence,
            epistemic_confidence: confidence as f64,
            consciousness_phi: self.current_phi,
            unified_free_energy: 1.0 - confidence as f64, // Higher certainty = lower free energy
            execution_strategy: strategy.clone(),
            recommended_strategy: strategy,
            clarifying_questions,
            intent_classification: Some(intent_classification),
            primitive_tiers,
            primitives,
        }
    }

    /// Map a semantic intent and input text to concrete executable primitives.
    fn ground_to_primitives(input: &str, classification: &IntentClassification) -> Vec<String> {
        let mut primitives = Vec::new();
        let lower = input.to_lowercase();

        // 1. Detection of Fix/Update/Optimize intent
        if classification.category == IntentCategory::NixOS
            || classification.category == IntentCategory::Programming
        {
            let evolution_keywords = [
                "fix",
                "broken",
                "missing",
                "repair",
                "optimize",
                "implementation",
                "propose",
                "improve",
                "refactor",
            ];
            if evolution_keywords.iter().any(|k| lower.contains(k)) || lower.contains("look at") {
                primitives.push("READ".to_string());
                primitives.push("WRITE".to_string());
                if lower.contains("nix") {
                    primitives.push("NIX_BUILD".to_string());
                }
            } else {
                primitives.push("CARGO_CHECK".to_string());
            }
        }

        // 2. Explicit action keywords (only add if not already present)
        if (lower.contains("build") || lower.contains("compile"))
            && !primitives.contains(&"NIX_BUILD".to_string())
        {
            primitives.push("NIX_BUILD".to_string());
        }
        if (lower.contains("commit") || lower.contains("persist"))
            && !primitives.contains(&"GIT_COMMIT".to_string())
        {
            primitives.push("GIT_COMMIT".to_string());
        }
        if (lower.contains("search") || lower.contains("look up") || lower.contains("research"))
            && !primitives.contains(&"WEB_SEARCH".to_string())
        {
            primitives.push("WEB_SEARCH".to_string());
        }
        if (lower.contains("list") || lower.contains("ls"))
            && !primitives.contains(&"LIST".to_string())
        {
            primitives.push("LIST".to_string());
        }
        if (lower.contains("read") || lower.contains("cat"))
            && !primitives.contains(&"READ".to_string())
        {
            primitives.push("READ".to_string());
        }

        // 3. HAL Primitives grounding
        if lower.contains("power")
            || lower.contains("sensor")
            || lower.contains("voltage")
            || lower.contains("ina219")
        {
            primitives.push("READ_SENSOR".to_string());
        }
        if lower.contains("servo")
            || lower.contains("joint")
            || lower.contains("move")
            || lower.contains("actuator")
        {
            primitives.push("WRITE_SERVO".to_string());
        }

        if lower.contains("share")
            || lower.contains("broadcast")
            || lower.contains("tell the others")
            || lower.contains("swarm")
            || lower.contains("gossip")
        {
            primitives.push("SWARM_GOSSIP".to_string());
        }

        if lower.contains("wasm")
            || lower.contains("sandbox")
            || lower.contains("verify")
            || lower.contains("binary")
        {
            primitives.push("WASM_VERIFY".to_string());
        }

        primitives
    }

    /// Map a semantic intent classification to likely active primitive tiers.
    ///
    /// This is the "grounding step" that connects language understanding to
    /// the 9-tier ontological primitive system. Each intent category activates
    /// a characteristic set of primitive tiers.
    fn ground_to_primitive_tiers(classification: &IntentClassification) -> Vec<String> {
        let mut tiers = Vec::new();

        // Every input activates NSM (Natural Semantic Metalanguage) primitives
        tiers.push("NSM".to_string());

        match classification.category {
            IntentCategory::NixOS => {
                tiers.push("Strategic".to_string()); // Planning, goal-directed action
                tiers.push("Compositional".to_string()); // System composition
                tiers.push("Temporal".to_string()); // Sequencing (build steps)
            }
            IntentCategory::Programming => {
                tiers.push("Strategic".to_string()); // Problem decomposition
                tiers.push("Compositional".to_string()); // Code composition
                tiers.push("Mathematical".to_string()); // Logic, algorithms
                tiers.push("MetaCognitive".to_string()); // Debugging, reflection
            }
            IntentCategory::Math => {
                tiers.push("Mathematical".to_string()); // Core math primitives
                tiers.push("Physical".to_string()); // Quantities, units
                tiers.push("Geometric".to_string()); // Spatial reasoning
            }
            IntentCategory::SystemAdmin => {
                tiers.push("Strategic".to_string()); // System management
                tiers.push("Temporal".to_string()); // Process sequencing
                tiers.push("Physical".to_string()); // Hardware, resources
            }
            IntentCategory::General => {
                tiers.push("MetaCognitive".to_string()); // General reasoning
                tiers.push("Consciousness".to_string()); // Awareness, understanding
            }
        }

        // High-confidence classifications also activate consciousness tier
        if classification.confidence > 0.7 && !tiers.contains(&"Consciousness".to_string()) {
            tiers.push("Consciousness".to_string());
        }

        tiers
    }

    /// Determine consciousness quadrant and execution strategy based on Φ and confidence
    fn determine_quadrant_and_strategy(
        &self,
        phi: f64,
        confidence: f64,
    ) -> (ConsciousnessQuadrant, ExecutionStrategy) {
        const PHI_THRESHOLD: f64 = 0.5;
        const CONF_THRESHOLD: f64 = 0.5;

        if phi >= PHI_THRESHOLD && confidence >= CONF_THRESHOLD {
            // High Φ + High Confidence: Confident
            (
                ConsciousnessQuadrant::Analytical,
                ExecutionStrategy::confident(),
            )
        } else if phi < PHI_THRESHOLD && confidence >= CONF_THRESHOLD {
            // Low Φ + High Confidence: Autopilot
            (
                ConsciousnessQuadrant::Intuitive,
                ExecutionStrategy::autopilot(),
            )
        } else if phi >= PHI_THRESHOLD && confidence < CONF_THRESHOLD {
            // High Φ + Low Confidence: Curious
            (
                ConsciousnessQuadrant::Emotional,
                ExecutionStrategy::safe_default(),
            )
        } else {
            // Low Φ + Low Confidence: Lost
            (ConsciousnessQuadrant::Somatic, ExecutionStrategy::lost())
        }
    }

    /// Extract NixOS intent from text
    fn extract_nixos_intent(&self, text: &str) -> NixOSIntent {
        let lower = text.to_lowercase();

        if lower.contains("install") {
            NixOSIntent::Install
        } else if lower.contains("remove") || lower.contains("uninstall") {
            NixOSIntent::Remove
        } else if lower.contains("search") || lower.contains("find") {
            NixOSIntent::Search
        } else if lower.contains("update") || lower.contains("upgrade") {
            NixOSIntent::Upgrade
        } else if lower.contains("configure") || lower.contains("setup") {
            NixOSIntent::Configure
        } else if lower.contains("list") || lower.contains("show") {
            NixOSIntent::List
        } else if lower.contains("garbage") || lower.contains("gc") {
            NixOSIntent::GarbageCollect
        } else if lower.contains("rollback") {
            NixOSIntent::Rollback
        } else if lower.contains("flake") {
            NixOSIntent::FlakeOp
        } else if lower.contains("build") {
            NixOSIntent::Build
        } else if lower.contains("switch") {
            NixOSIntent::Switch
        } else if lower.contains("info") || lower.contains("about") {
            NixOSIntent::Info
        } else {
            NixOSIntent::Unknown
        }
    }

    /// Extract entities (packages, options) from text
    fn extract_entities(&self, text: &str) -> Vec<String> {
        // Simple word extraction - would use NLP in production
        text.split_whitespace()
            .filter(|w| {
                w.len() > 2
                    && ![
                        "the", "and", "for", "with", "from", "install", "remove", "search",
                    ]
                    .contains(w)
            })
            .map(|s| s.to_string())
            .collect()
    }

    /// Diagnose an error with consciousness awareness
    pub fn diagnose_error(&self, error: &str) -> ErrorDiagnosis {
        self.diagnoser.diagnose(error)
    }

    /// Get current consciousness state
    pub fn consciousness_state(&self) -> ConsciousnessStateLevel {
        self.state
    }

    /// Set consciousness state
    pub fn set_consciousness_state(&mut self, state: ConsciousnessStateLevel) {
        self.state = state;
    }

    /// Process feedback to improve (legacy method)
    pub fn process_feedback(&mut self, _feedback: ActionOutcomeFeedback) {
        // Would update internal models based on feedback
        // Stub implementation
    }

    /// Receive action outcome feedback from the pipeline (integration module method)
    pub fn receive_action_outcome(&mut self, feedback: ActionOutcomeFeedback) {
        // Update Φ based on action success
        if feedback.action_succeeded {
            self.current_phi = (self.current_phi * 1.05).min(1.0);
        } else {
            self.current_phi = (self.current_phi * 0.95).max(0.1);
        }

        // Update confidence based on user feedback
        if let Some(positive) = feedback.user_feedback {
            if positive {
                self.current_confidence = (self.current_confidence * 1.1).min(1.0);
            } else {
                self.current_confidence = (self.current_confidence * 0.9).max(0.1);
            }
        }
    }

    /// Create feedback from execution result (integration module method)
    pub fn create_feedback_from_execution(
        &self,
        success: bool,
        phi: f64,
        error: Option<String>,
        was_dry_run: bool,
    ) -> Option<ActionOutcomeFeedback> {
        Some(ActionOutcomeFeedback {
            original_input: self.last_input.clone(),
            decided_in_quadrant: self.last_quadrant,
            strategy_used: ExecutionStrategyType::Direct,
            phi_at_decision: self.current_phi,
            confidence_at_decision: self.current_confidence,
            action_succeeded: success,
            phi_after: phi,
            error_message: error.clone(),
            was_dry_run,
            user_feedback: None,
            success,
            execution_time_ms: 0,
            user_satisfaction: None,
            errors: error.into_iter().collect(),
            lessons: Vec::new(),
        })
    }

    /// Get current Φ level
    pub fn phi(&self) -> f64 {
        self.current_phi
    }

    /// Set Φ level (for calibration)
    pub fn set_phi(&mut self, phi: f64) {
        self.current_phi = phi.clamp(0.0, 1.0);
    }
}

impl Default for ConsciousnessLanguageCore {
    fn default() -> Self {
        Self::new(ConsciousnessLanguageConfig::default())
    }
}

// Note: Types defined above are automatically public since they're
// declared with `pub`. No need for re-export.

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // NixErrorDiagnoser tests
    // ========================================================================

    #[test]
    fn nix_error_diagnoser_new_creates_successfully() {
        let diagnoser = NixErrorDiagnoser::new();
        // The diagnoser should have compiled its regex patterns without panicking.
        // Default impl should also work identically.
        let _default = NixErrorDiagnoser::default();
        assert!(!diagnoser.patterns.is_empty());
    }

    #[test]
    fn diagnose_classifies_missing_attribute() {
        let diagnoser = NixErrorDiagnoser::new();
        let diagnosis = diagnoser
            .diagnose("error: attribute 'foo' missing at /etc/nixos/configuration.nix:42:5");
        assert_eq!(diagnosis.category, NixErrorCategory::Evaluation);
        assert!(diagnosis.confidence > 0.0 && diagnosis.confidence.is_finite());
        assert!(!diagnosis.explanation.is_empty());
        // Location extraction should pick up the file path
        assert!(diagnosis.location.is_some());
        let loc = diagnosis.location.unwrap();
        assert!(loc.contains("/etc/nixos/configuration.nix"));
    }

    #[test]
    fn diagnose_classifies_undefined_variable() {
        let diagnoser = NixErrorDiagnoser::new();
        let diagnosis = diagnoser.diagnose("error: undefined variable 'pkgs'");
        assert_eq!(diagnosis.category, NixErrorCategory::Evaluation);
        assert!(diagnosis.confidence >= 0.5);
    }

    #[test]
    fn diagnose_classifies_build_failure() {
        let diagnoser = NixErrorDiagnoser::new();
        let diagnosis =
            diagnoser.diagnose("builder for '/nix/store/abc-pkg.drv' failed with exit code 1");
        assert_eq!(diagnosis.category, NixErrorCategory::Build);
    }

    #[test]
    fn diagnose_classifies_permission_error() {
        let diagnoser = NixErrorDiagnoser::new();
        let diagnosis = diagnoser.diagnose("error: permission denied while accessing /nix/store");
        assert_eq!(diagnosis.category, NixErrorCategory::Permission);
    }

    #[test]
    fn diagnose_returns_unknown_for_unrecognised_text() {
        let diagnoser = NixErrorDiagnoser::new();
        let diagnosis = diagnoser.diagnose("something completely unrelated happened");
        assert_eq!(diagnosis.category, NixErrorCategory::Unknown);
        // Unknown errors should have low confidence
        assert!(diagnosis.confidence < 0.5);
        assert!(!diagnosis.explanation.is_empty());
    }

    // ========================================================================
    // ConsciousnessLanguageCore tests
    // ========================================================================

    #[test]
    fn consciousness_language_core_default_construction() {
        let core = ConsciousnessLanguageCore::default();
        assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Active);
        assert!(core.phi().is_finite());
        assert!(core.phi() >= 0.0 && core.phi() <= 1.0);
    }

    #[test]
    fn consciousness_state_get_and_set() {
        let mut core = ConsciousnessLanguageCore::default();
        assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Active);

        core.set_consciousness_state(ConsciousnessStateLevel::Reflective);
        assert_eq!(
            core.consciousness_state(),
            ConsciousnessStateLevel::Reflective
        );

        core.set_consciousness_state(ConsciousnessStateLevel::Dormant);
        assert_eq!(core.consciousness_state(), ConsciousnessStateLevel::Dormant);
    }

    #[test]
    fn phi_get_set_and_clamping() {
        let mut core = ConsciousnessLanguageCore::default();
        let initial_phi = core.phi();
        assert!(initial_phi.is_finite());

        core.set_phi(0.8);
        assert!((core.phi() - 0.8).abs() < f64::EPSILON);

        // Values above 1.0 should clamp to 1.0
        core.set_phi(5.0);
        assert!((core.phi() - 1.0).abs() < f64::EPSILON);

        // Values below 0.0 should clamp to 0.0
        core.set_phi(-2.0);
        assert!(core.phi().abs() < f64::EPSILON);
    }

    #[test]
    fn understand_produces_finite_confidence() {
        let mut core = ConsciousnessLanguageCore::default();
        let result = core.understand("install firefox on NixOS");

        assert!(result.confidence.is_finite());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.epistemic_confidence.is_finite());
        assert!(result.consciousness_phi.is_finite());
        assert!(result.unified_free_energy.is_finite());

        // An install-related query should be classified as Install
        assert_eq!(result.nixos.intent, NixOSIntent::Install);
        assert!(!result.nixos.description.is_empty());
    }

    #[test]
    fn understand_unknown_input_yields_low_confidence_and_clarifying_questions() {
        let mut core = ConsciousnessLanguageCore::default();
        // Gibberish that does not match any NixOS keyword
        let result = core.understand("xyzzy plugh");

        assert_eq!(result.nixos.intent, NixOSIntent::Unknown);
        // Unknown intent produces confidence 0.3, which is below the default
        // confidence_threshold (0.6), so clarifying questions should be generated.
        assert!(!result.clarifying_questions.is_empty());
    }

    #[test]
    fn create_feedback_from_execution_produces_valid_feedback() {
        let mut core = ConsciousnessLanguageCore::default();
        // Run understand first so last_input / last_quadrant are populated
        let _result = core.understand("build my flake");

        let feedback = core.create_feedback_from_execution(true, 0.7, None, false);
        assert!(feedback.is_some());
        let fb = feedback.unwrap();
        assert!(fb.action_succeeded);
        assert!(fb.success);
        assert!(!fb.was_dry_run);
        assert!(fb.error_message.is_none());
        assert!(fb.errors.is_empty());
        assert!((fb.phi_after - 0.7).abs() < f64::EPSILON);
        // original_input should reflect the last understand() call
        assert!(fb.original_input.contains("build my flake"));

        // Also test failure case
        let feedback_err =
            core.create_feedback_from_execution(false, 0.3, Some("boom".to_string()), true);
        let fb_err = feedback_err.unwrap();
        assert!(!fb_err.action_succeeded);
        assert!(fb_err.was_dry_run);
        assert_eq!(fb_err.error_message.as_deref(), Some("boom"));
        assert_eq!(fb_err.errors, vec!["boom".to_string()]);
    }
}
