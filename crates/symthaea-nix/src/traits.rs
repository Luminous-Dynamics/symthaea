//! Local trait and type definitions for the symthaea-nix crate.
//!
//! These are standalone definitions that mirror the main symthaea crate's types,
//! allowing the NixOS-specific modules to compile independently without pulling
//! in the full symthaea-core dependency graph.

use std::collections::HashMap;
use tree_sitter::Tree;

// ============================================================================
// Code Parsing
// ============================================================================

/// Trait for language-specific code parsers.
pub trait CodeParser: Send + Sync {
    fn language_name(&self) -> &str;
    fn parse(&mut self, source: &str) -> Result<ParsedCode, CodeDiagnostic>;
    fn extract_entities(&self, parsed: &ParsedCode) -> Vec<CodeEntity>;
    fn detect_diagnostics(&self, parsed: &ParsedCode) -> Vec<CodeDiagnostic>;
    fn file_extensions(&self) -> &[&str];
}

/// The result of parsing a source file.
#[derive(Debug, Clone)]
pub struct ParsedCode {
    pub source: String,
    pub language: String,
    pub entities: Vec<CodeEntity>,
    pub diagnostics: Vec<CodeDiagnostic>,
    tree: Option<Tree>,
}

impl ParsedCode {
    pub fn new(source: &str, language: &str) -> Self {
        Self {
            source: source.to_string(),
            language: language.to_string(),
            entities: Vec::new(),
            diagnostics: Vec::new(),
            tree: None,
        }
    }

    pub fn with_tree(mut self, tree: Tree) -> Self {
        self.tree = Some(tree);
        self
    }

    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<&CodeEntity> {
        self.entities.iter().filter(|e| e.kind == kind).collect()
    }
}

// ============================================================================
// Code Entities
// ============================================================================

/// A structural entity extracted from parsed code.
#[derive(Debug, Clone)]
pub struct CodeEntity {
    pub kind: EntityKind,
    pub name: String,
    pub span: Span,
    pub source_text: Option<String>,
    pub children: Vec<CodeEntity>,
    pub annotations: HashMap<String, String>,
}

impl CodeEntity {
    pub fn new(kind: EntityKind, name: &str, span: Span) -> Self {
        Self {
            kind,
            name: name.to_string(),
            span,
            source_text: None,
            children: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    pub fn with_source(mut self, source: String) -> Self {
        self.source_text = Some(source);
        self
    }

    pub fn with_annotation(mut self, key: &str, value: &str) -> Self {
        self.annotations.insert(key.to_string(), value.to_string());
        self
    }
}

/// The kind of a code entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Module,
    Variable,
    Constant,
    Import,
    Binding,
    AttrSet,
    WithExpression,
    FlakeInput,
    FlakeOutput,
}

/// A byte-and-line span within a source file.
#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

/// A diagnostic message produced during parsing or analysis.
#[derive(Debug, Clone)]
pub struct CodeDiagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub span: Option<Span>,
}

/// Severity level for code diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

// ============================================================================
// Domain Plugin
// ============================================================================

/// Trait for domain-specific plugins that provide entity extraction,
/// intent prototypes, prompts, and diagnostic capabilities.
pub trait DomainPlugin: Send + Sync {
    fn domain_name(&self) -> &str;
    fn extract_entities(&self, text: &str) -> Vec<Entity>;
    fn intent_prototypes(&self) -> IntentPrototypes {
        IntentPrototypes::default()
    }
    fn prompts(&self) -> DomainPrompts {
        DomainPrompts::default()
    }
    fn diagnose_error(&self, _error_text: &str) -> Option<ErrorDiagnosis> {
        None
    }
    fn validate_input(&self, _input: &str) -> ValidationResult {
        ValidationResult::valid()
    }
    fn is_in_domain(&self, _topic: &str) -> f64 {
        0.5
    }
    fn vocabulary(&self) -> Vec<String> {
        vec![]
    }
    fn preprocess(&self, input: &str) -> String {
        input.to_string()
    }
    fn suggest_actions(&self, _context: &str) -> Vec<String> {
        vec![]
    }
}

/// A domain-level entity extracted from text.
#[derive(Debug, Clone)]
pub struct Entity {
    pub entity_type: String,
    pub value: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
    pub metadata: HashMap<String, String>,
}

impl Entity {
    pub fn new(entity_type: &str, value: &str, start: usize, end: usize) -> Self {
        Self {
            entity_type: entity_type.to_string(),
            value: value.to_string(),
            start,
            end,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Prototype sentences for intent classification.
#[derive(Debug, Clone, Default)]
pub struct IntentPrototypes {
    pub greeting: Vec<String>,
    pub question: Vec<String>,
    pub command: Vec<String>,
    pub complaint: Vec<String>,
    pub feedback: Vec<String>,
    pub custom: HashMap<String, Vec<String>>,
}

/// Domain-specific prompt templates.
#[derive(Debug, Clone, Default)]
pub struct DomainPrompts {
    pub system: String,
    pub clarification: String,
    pub action_confirm: String,
    pub error_explain: String,
    pub out_of_domain: String,
}

/// A structured diagnosis of an error.
#[derive(Debug, Clone)]
pub struct ErrorDiagnosis {
    pub category: String,
    pub error_type: String,
    pub description: String,
    pub suggestion: Option<String>,
    pub confidence: f64,
    pub risk_level: RiskLevel,
    pub location: Option<ErrorLocation>,
}

/// Risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Location information attached to an error diagnosis.
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub context: Option<String>,
}

/// The result of validating an input string.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

// ============================================================================
// Consciousness Types
// ============================================================================

/// Classification of actions by their impact level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    BasicQuery,
    StateModifying,
    SystemCritical,
    Irreversible,
}

/// Phi-based thresholds that gate action execution by consciousness level.
#[derive(Debug, Clone)]
pub struct ConsciousnessThresholds {
    pub basic_query: f32,
    pub state_modifying: f32,
    pub system_critical: f32,
    pub irreversible: f32,
}

impl Default for ConsciousnessThresholds {
    fn default() -> Self {
        Self {
            basic_query: 0.2,
            state_modifying: 0.3,
            system_critical: 0.4,
            irreversible: 0.6,
        }
    }
}

impl ConsciousnessThresholds {
    pub fn threshold_for(&self, action: ActionType) -> f32 {
        match action {
            ActionType::BasicQuery => self.basic_query,
            ActionType::StateModifying => self.state_modifying,
            ActionType::SystemCritical => self.system_critical,
            ActionType::Irreversible => self.irreversible,
        }
    }

    pub fn allows_action(&self, action: ActionType, phi: f32) -> bool {
        phi >= self.threshold_for(action)
    }
}

/// Maps a phi value to a qualitative confidence level with a recommendation.
pub struct PhiAwareScoring;

impl PhiAwareScoring {
    pub fn confidence_level(phi: f32) -> ConfidenceLevel {
        if phi >= 0.6 {
            ConfidenceLevel::High {
                phi,
                recommendation: "Proceed with confidence".to_string(),
            }
        } else if phi >= 0.4 {
            ConfidenceLevel::Medium {
                phi,
                recommendation: "Proceed with standard caution".to_string(),
            }
        } else if phi >= 0.2 {
            ConfidenceLevel::Low {
                phi,
                recommendation: "Consider user confirmation".to_string(),
            }
        } else {
            ConfidenceLevel::VeryLow {
                phi,
                recommendation: "Require explicit confirmation".to_string(),
            }
        }
    }
}

/// Qualitative confidence level derived from a phi value.
#[derive(Debug, Clone)]
pub enum ConfidenceLevel {
    High { phi: f32, recommendation: String },
    Medium { phi: f32, recommendation: String },
    Low { phi: f32, recommendation: String },
    VeryLow { phi: f32, recommendation: String },
}

impl ConfidenceLevel {
    pub fn recommendation(&self) -> &str {
        match self {
            Self::High { recommendation, .. } => recommendation,
            Self::Medium { recommendation, .. } => recommendation,
            Self::Low { recommendation, .. } => recommendation,
            Self::VeryLow { recommendation, .. } => recommendation,
        }
    }
}

// ============================================================================
// Causal Discovery Engine (Stub)
// ============================================================================

/// Stub for causal discovery engine - will be replaced with symthaea-core's CausalMind.
pub struct CausalDiscoveryEngine {
    _seed: u64,
}

/// Direction of a discovered causal relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CausalDirection {
    Forward,
    Backward,
}

impl CausalDiscoveryEngine {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }

    /// Predict causal direction between two time series with a confidence score.
    ///
    /// This is a simple correlation-based stub. The real implementation lives
    /// in symthaea-core's `CausalMind`.
    pub fn predict_with_confidence(&mut self, x: &[f64], y: &[f64]) -> (CausalDirection, f64) {
        let n = x.len().min(y.len());
        if n < 2 {
            return (CausalDirection::Forward, 0.5);
        }
        let mean_x: f64 = x.iter().take(n).sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().take(n).sum::<f64>() / n as f64;
        let cov: f64 = x
            .iter()
            .take(n)
            .zip(y.iter().take(n))
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / n as f64;
        let direction = if cov >= 0.0 {
            CausalDirection::Forward
        } else {
            CausalDirection::Backward
        };
        let confidence = (cov.abs() / (mean_x.abs().max(1.0) * mean_y.abs().max(1.0))).min(1.0);
        (direction, confidence)
    }
}
