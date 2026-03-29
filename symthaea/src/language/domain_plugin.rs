// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Domain Plugin Trait - Enables domain-agnostic language processing
//!
//! This module provides the abstraction layer that allows Symthaea's language
//! system to work with any domain, not just NixOS.
//!
//! # Architecture
//!
//! The core language system (intent classification, emotional analysis, semantic
//! encoding, conversation memory) is domain-agnostic. Domain-specific behavior
//! is provided through plugins that implement the `DomainPlugin` trait.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │           SYMTHAEA LANGUAGE CORE (Domain-Agnostic)      │
//! │  • Intent Classification (HDC-based)                    │
//! │  • Semantic Encoding (Qwen3 + HDC)                      │
//! │  • Emotional Analysis                                   │
//! │  • Conversation Memory                                  │
//! │  • Φ Monitoring                                         │
//! └───────────────────────────┬─────────────────────────────┘
//!                             │
//!                     ┌───────▼───────┐
//!                     │ DomainPlugin  │
//!                     └───────┬───────┘
//!                             │
//!     ┌───────────────────────┼───────────────────────────┐
//!     │                       │                           │
//!     ▼                       ▼                           ▼
//! ┌────────┐            ┌────────────┐            ┌────────────┐
//! │ NixOS  │            │ Customer   │            │ Medical    │
//! │ Plugin │            │ Support    │            │ Assistant  │
//! └────────┘            └────────────┘            └────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use symthaea::language::domain_plugin::{DomainPlugin, Entity, ErrorDiagnosis};
//!
//! struct MyDomainPlugin;
//!
//! impl DomainPlugin for MyDomainPlugin {
//!     fn domain_name(&self) -> &str { "my-domain" }
//!     fn extract_entities(&self, text: &str) -> Vec<Entity> { vec![] }
//!     // ... implement other methods
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::mind::structured_thought::EpistemicCube;

// ============================================================================
// CORE TYPES
// ============================================================================

/// An entity extracted from user input
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    /// Type of entity (domain-specific)
    pub entity_type: String,
    /// The extracted value
    pub value: String,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity
    pub fn new(
        entity_type: impl Into<String>,
        value: impl Into<String>,
        start: usize,
        end: usize,
    ) -> Self {
        Self {
            entity_type: entity_type.into(),
            value: value.into(),
            start,
            end,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// An error diagnosis result
#[derive(Debug, Clone)]
pub struct ErrorDiagnosis {
    /// Error category
    pub category: String,
    /// Specific error type
    pub error_type: String,
    /// Human-readable description
    pub description: String,
    /// Suggested fix (if available)
    pub suggestion: Option<String>,
    /// Confidence in diagnosis (0.0 - 1.0)
    pub confidence: f64,
    /// Risk level of the error
    pub risk_level: RiskLevel,
    /// Location in source (if applicable)
    pub location: Option<ErrorLocation>,
}

/// Risk level for errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Safe to fix automatically
    Safe,
    /// Low risk, minor impact
    Low,
    /// Medium risk, user should review
    Medium,
    /// High risk, could cause problems
    High,
    /// Critical, could cause data loss or system issues
    Critical,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "Safe"),
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Location of an error in source
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// File path (if applicable)
    pub file: Option<String>,
    /// Line number (1-based)
    pub line: Option<usize>,
    /// Column number (1-based)
    pub column: Option<usize>,
    /// Relevant context snippet
    pub context: Option<String>,
}

/// Intent prototypes for a domain
#[derive(Debug, Clone)]
pub struct IntentPrototypes {
    /// Words/phrases for greeting intent
    pub greeting: Vec<String>,
    /// Words/phrases for question intent
    pub question: Vec<String>,
    /// Words/phrases for command intent
    pub command: Vec<String>,
    /// Words/phrases for complaint/issue intent
    pub complaint: Vec<String>,
    /// Words/phrases for feedback/praise intent
    pub feedback: Vec<String>,
    /// Custom intents (name -> words)
    pub custom: HashMap<String, Vec<String>>,
}

impl Default for IntentPrototypes {
    fn default() -> Self {
        Self {
            greeting: vec![
                "hello",
                "hi",
                "hey",
                "greetings",
                "good morning",
                "good afternoon",
                "good evening",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            question: vec![
                "what", "why", "how", "when", "where", "which", "can", "could", "would", "is",
                "are", "do", "does",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            command: vec![
                "install",
                "remove",
                "update",
                "configure",
                "set",
                "enable",
                "disable",
                "start",
                "stop",
                "create",
                "delete",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            complaint: vec![
                "problem",
                "issue",
                "error",
                "broken",
                "not working",
                "failed",
                "doesn't work",
                "can't",
                "unable",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            feedback: vec![
                "thanks",
                "thank you",
                "great",
                "awesome",
                "perfect",
                "excellent",
                "good",
                "nice",
                "helpful",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            custom: HashMap::new(),
        }
    }
}

/// Domain-specific prompt templates
#[derive(Debug, Clone)]
pub struct DomainPrompts {
    /// System prompt for LLM
    pub system: String,
    /// Clarification prompt template
    pub clarification: String,
    /// Action confirmation template
    pub action_confirm: String,
    /// Error explanation template
    pub error_explain: String,
    /// Out-of-domain response template
    pub out_of_domain: String,
}

impl Default for DomainPrompts {
    fn default() -> Self {
        Self {
            system: "You are a helpful assistant.".to_string(),
            clarification: "Could you please clarify what you mean by '{}'?".to_string(),
            action_confirm: "I'll help you with that. To confirm: {}".to_string(),
            error_explain: "I encountered an issue: {}".to_string(),
            out_of_domain: "I'm not sure I can help with that. {}".to_string(),
        }
    }
}

/// Validation result for domain-specific input
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is the input valid?
    pub valid: bool,
    /// Validation errors (if any)
    pub errors: Vec<String>,
    /// Warnings (non-blocking issues)
    pub warnings: Vec<String>,
    /// Suggested corrections
    pub suggestions: Vec<String>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            valid: true,
            errors: vec![],
            warnings: vec![],
            suggestions: vec![],
        }
    }

    /// Create an invalid result with errors
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: vec![],
            suggestions: vec![],
        }
    }
}

// ============================================================================
// COMPUTED RESULT
// ============================================================================

/// Rich metadata returned by a domain plugin's `compute()` method.
///
/// Carries the deterministic answer along with epistemic classification
/// and proof metadata for principled epistemic status derivation.
#[derive(Debug, Clone)]
pub struct ComputedResult {
    /// The deterministic answer string (e.g., "2 + 2 = 4")
    pub answer: String,
    /// 3D epistemic classification from the Mycelix Epistemic Charter
    pub cube: EpistemicCube,
    /// Ψ — Consciousness estimate from the HDC proof
    pub psi: f64,
    /// Whether a formal proof is available for this result
    pub proof_available: bool,
}

// ============================================================================
// DOMAIN PLUGIN TRAIT
// ============================================================================

/// Trait for domain-specific language processing plugins
///
/// Implement this trait to add support for a new domain to Symthaea's
/// language processing system.
pub trait DomainPlugin: Send + Sync {
    // ========================================================================
    // REQUIRED METHODS
    // ========================================================================

    /// Get the domain name (e.g., "nixos", "customer-support", "medical")
    fn domain_name(&self) -> &str;

    /// Extract domain-specific entities from text
    ///
    /// # Arguments
    /// * `text` - The input text to analyze
    ///
    /// # Returns
    /// A list of extracted entities
    fn extract_entities(&self, text: &str) -> Vec<Entity>;

    // ========================================================================
    // OPTIONAL METHODS (with default implementations)
    // ========================================================================

    /// Get intent prototypes for this domain
    ///
    /// Override to customize intent classification vocabulary
    fn intent_prototypes(&self) -> IntentPrototypes {
        IntentPrototypes::default()
    }

    /// Get prompt templates for this domain
    ///
    /// Override to customize LLM prompts
    fn prompts(&self) -> DomainPrompts {
        DomainPrompts::default()
    }

    /// Diagnose an error message
    ///
    /// # Arguments
    /// * `error_text` - The error message to diagnose
    ///
    /// # Returns
    /// Diagnosis if the error is recognized, None otherwise
    fn diagnose_error(&self, _error_text: &str) -> Option<ErrorDiagnosis> {
        None
    }

    /// Validate domain-specific input
    ///
    /// # Arguments
    /// * `input` - The input to validate
    ///
    /// # Returns
    /// Validation result
    fn validate_input(&self, _input: &str) -> ValidationResult {
        ValidationResult::valid()
    }

    /// Check if a topic is within this domain
    ///
    /// # Arguments
    /// * `topic` - The topic to check
    ///
    /// # Returns
    /// Confidence that the topic is in-domain (0.0 - 1.0)
    fn is_in_domain(&self, _topic: &str) -> f64 {
        0.5 // Neutral by default
    }

    /// Get domain-specific vocabulary for semantic encoding
    ///
    /// Override to add domain-specific words that should be recognized
    fn vocabulary(&self) -> Vec<String> {
        vec![]
    }

    /// Transform user input before processing
    ///
    /// Override to preprocess domain-specific syntax
    fn preprocess(&self, input: &str) -> String {
        input.to_string()
    }

    /// Transform output before presenting to user
    ///
    /// Override to postprocess domain-specific formatting
    fn postprocess(&self, output: &str) -> String {
        output.to_string()
    }

    /// Get suggested actions for a given context
    ///
    /// # Arguments
    /// * `context` - Current conversation context
    ///
    /// # Returns
    /// List of suggested actions
    fn suggest_actions(&self, _context: &str) -> Vec<String> {
        vec![]
    }

    /// Compute a deterministic answer for the given input and entities.
    ///
    /// Domain plugins can override this to provide Rust-computed answers
    /// that bypass LLM generation entirely (e.g., arithmetic results).
    /// Returns a `ComputedResult` carrying the answer, epistemic cube
    /// classification, Φ value, and proof availability.
    ///
    /// # Returns
    /// `Some(ComputedResult)` if a deterministic answer was computed,
    /// `None` if the query requires LLM translation.
    fn compute(&self, _input: &str, _entities: &[Entity]) -> Option<ComputedResult> {
        None
    }
}

// ============================================================================
// BUILT-IN PLUGINS
// ============================================================================

/// Generic domain plugin with no specialization
pub struct GenericPlugin;

impl DomainPlugin for GenericPlugin {
    fn domain_name(&self) -> &str {
        "generic"
    }

    fn extract_entities(&self, _text: &str) -> Vec<Entity> {
        vec![]
    }
}

// ============================================================================
// PLUGIN REGISTRY
// ============================================================================

/// Registry for domain plugins
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn DomainPlugin>>,
    default_plugin: String,
}

impl PluginRegistry {
    /// Create a new registry with the generic plugin as default
    pub fn new() -> Self {
        let mut registry = Self {
            plugins: HashMap::new(),
            default_plugin: "generic".to_string(),
        };
        registry.register(Box::new(GenericPlugin));
        registry
    }

    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn DomainPlugin>) {
        let name = plugin.domain_name().to_string();
        self.plugins.insert(name, plugin);
    }

    /// Set the default plugin
    pub fn set_default(&mut self, name: &str) -> Result<(), crate::errors::SymthaeaError> {
        if self.plugins.contains_key(name) {
            self.default_plugin = name.to_string();
            Ok(())
        } else {
            Err(crate::errors::SymthaeaError::Language(format!(
                "Plugin '{name}' not found"
            )))
        }
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Option<&dyn DomainPlugin> {
        self.plugins.get(name).map(|p| p.as_ref())
    }

    /// Get the default plugin
    pub fn default_plugin(&self) -> &dyn DomainPlugin {
        self.plugins
            .get(&self.default_plugin)
            .map(|p| p.as_ref())
            .expect("default_plugin key must exist in plugins map")
    }

    /// List all registered plugins
    pub fn list(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }

    /// Minimum score a plugin must achieve to be selected as the domain.
    /// Prevents low-confidence false positives (e.g., "Atlantis" matching random plugins).
    const MIN_DOMAIN_THRESHOLD: f64 = 0.3;

    /// Auto-detect domain from input
    pub fn detect_domain(&self, input: &str) -> &str {
        let mut best_match = &self.default_plugin;
        let mut best_score = 0.0;

        for (name, plugin) in &self.plugins {
            if name == &self.default_plugin {
                continue; // Don't score the generic/default plugin
            }
            let score = plugin.is_in_domain(input);
            if score > best_score && score >= Self::MIN_DOMAIN_THRESHOLD {
                best_score = score;
                best_match = name;
            }
        }

        best_match
    }

    /// Create a registry pre-loaded with all built-in domain plugins
    ///
    /// Registers: generic (default), nixos, mathematics, programming,
    /// general-assistant, code-assistant, research, code (if feature enabled)
    pub fn with_builtins() -> Self {
        let mut registry = Self::new(); // starts with GenericPlugin
        registry.register(Box::new(super::nixos_plugin::NixOsPlugin));
        registry.register(Box::new(super::math_plugin::MathPlugin::new()));
        registry.register(Box::new(super::programming_plugin::ProgrammingPlugin));
        registry.register(Box::new(
            super::general_assistant_plugin::GeneralAssistantPlugin,
        ));
        registry.register(Box::new(super::code_assistant_plugin::CodeAssistantPlugin));
        registry.register(Box::new(super::research_plugin::ResearchPlugin));
        #[cfg(feature = "code_generation")]
        registry.register(Box::new(super::code_domain_plugin::CodeDomainPlugin));
        registry
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl DomainPlugin for TestPlugin {
        fn domain_name(&self) -> &str {
            "test"
        }

        fn extract_entities(&self, text: &str) -> Vec<Entity> {
            let mut entities = vec![];
            if text.contains("order") {
                entities.push(Entity::new("order_reference", "ORDER-123", 0, 9));
            }
            entities
        }

        fn is_in_domain(&self, topic: &str) -> f64 {
            if topic.contains("order") || topic.contains("shipping") {
                0.9
            } else {
                0.1
            }
        }
    }

    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("product", "Widget", 10, 16)
            .with_confidence(0.95)
            .with_metadata("category", "electronics");

        assert_eq!(entity.entity_type, "product");
        assert_eq!(entity.value, "Widget");
        assert_eq!(entity.confidence, 0.95);
        assert_eq!(
            entity.metadata.get("category"),
            Some(&"electronics".to_string())
        );
    }

    #[test]
    fn test_plugin_registry() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));
        registry.set_default("test").unwrap();

        assert!(registry.list().contains(&"test"));
        assert!(registry.list().contains(&"generic"));
        assert_eq!(registry.default_plugin().domain_name(), "test");
    }

    #[test]
    fn test_domain_detection() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));

        let detected = registry.detect_domain("I want to track my order");
        assert_eq!(detected, "test");

        let detected = registry.detect_domain("Hello world");
        assert_eq!(detected, "generic");
    }

    #[test]
    fn test_entity_extraction() {
        let plugin = TestPlugin;
        let entities = plugin.extract_entities("Check my order status");

        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "order_reference");
    }

    #[test]
    fn test_domain_detection_threshold_prevents_low_scores() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));

        // "Hello world" scores 0.1 for TestPlugin, below threshold 0.3
        let detected = registry.detect_domain("Hello world");
        assert_eq!(
            detected, "generic",
            "Low-score plugin should not be selected"
        );
    }

    #[test]
    fn test_domain_detection_threshold_allows_high_scores() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));

        // "I want to track my order" scores 0.9 for TestPlugin, above threshold
        let detected = registry.detect_domain("I want to track my order");
        assert_eq!(detected, "test", "High-score plugin should be selected");
    }
}
