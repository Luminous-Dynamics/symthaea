// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Structured Thought: The Language of Mind
//!
//! This module defines `StructuredThought` - the intermediate representation (IR)
//! of what the mind computes before translation into natural language.
//!
//! **Key Insight**: The LLM is NOT the brain - it's Broca's Area. The HDC+LTC mind
//! computes structured answers; the LLM merely translates those structures into
//! fluent natural language.
//!
//! This enables:
//! - Zero-hallucination reasoning (logic in Rust, deterministic)
//! - Transparent epistemic status (system knows what it doesn't know)
//! - Verifiable outputs (can check if LLM followed structured thought)
//! - Energy efficient (CPU reasoning, LLM only for fluency)

use serde::{Deserialize, Serialize};
use std::fmt;
use symthaea_core::hdc::relational_consciousness::{RelationMode, RelationshipStage};

// ============================================================================
// EPISTEMIC CUBE: 3-axis classification from Mycelix Epistemic Charter v2.0
// ============================================================================

/// Empirical axis: how verifiable is the claim?
///
/// E0 (opinion) → E4 (publicly reproducible proof)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ETier {
    E0,
    E1,
    E2,
    E3,
    E4,
}

/// Normative axis: how binding is the claim?
///
/// N0 (personal) → N3 (axiomatic truth like math)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NTier {
    N0,
    N1,
    N2,
    N3,
}

/// Materiality axis: how permanent is the claim?
///
/// M0 (ephemeral) → M3 (foundational)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MTier {
    M0,
    M1,
    M2,
    M3,
}

/// Harmonic axis: coherence/alignment with higher purpose
///
/// H0 (discordant) → H4 (transcendent)
/// Derived from phi (integrated information) and coherence scores.
///
/// This is LUCID's extension to the Mycelix Epistemic Charter to capture
/// consciousness-level metrics that are unique to Symthaea's analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum HTier {
    /// H0: Discordant - potentially harmful or very low coherence
    H0,
    /// H1: Neutral - no particular alignment (default)
    #[default]
    H1,
    /// H2: Resonant - moderate coherence
    H2,
    /// H3: Harmonic - high coherence
    H3,
    /// H4: Transcendent - maximum coherence, serves universal flourishing
    H4,
}

impl HTier {
    /// Derive HTier from phi (consciousness) and coherence scores
    pub fn from_phi_coherence(phi: f64, coherence: f64) -> Self {
        let combined = (phi + coherence) / 2.0;
        match combined {
            v if v < 0.125 => HTier::H0,
            v if v < 0.375 => HTier::H1,
            v if v < 0.625 => HTier::H2,
            v if v < 0.875 => HTier::H3,
            _ => HTier::H4,
        }
    }

    /// Convert to normalized f64 value
    pub fn to_f64(&self) -> f64 {
        match self {
            HTier::H0 => 0.0,
            HTier::H1 => 0.25,
            HTier::H2 => 0.5,
            HTier::H3 => 0.75,
            HTier::H4 => 1.0,
        }
    }
}

impl fmt::Display for HTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H0 => write!(f, "H0"),
            Self::H1 => write!(f, "H1"),
            Self::H2 => write!(f, "H2"),
            Self::H3 => write!(f, "H3"),
            Self::H4 => write!(f, "H4"),
        }
    }
}

impl fmt::Display for ETier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::E0 => write!(f, "E0"),
            Self::E1 => write!(f, "E1"),
            Self::E2 => write!(f, "E2"),
            Self::E3 => write!(f, "E3"),
            Self::E4 => write!(f, "E4"),
        }
    }
}

impl fmt::Display for NTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::N0 => write!(f, "N0"),
            Self::N1 => write!(f, "N1"),
            Self::N2 => write!(f, "N2"),
            Self::N3 => write!(f, "N3"),
        }
    }
}

impl fmt::Display for MTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::M0 => write!(f, "M0"),
            Self::M1 => write!(f, "M1"),
            Self::M2 => write!(f, "M2"),
            Self::M3 => write!(f, "M3"),
        }
    }
}

/// 3D/4D epistemic classification from the Mycelix Epistemic Charter v2.0.
///
/// Every claim is located in a cube with three core axes:
/// - **E-Axis (Empirical)**: E0 (opinion) → E4 (publicly reproducible)
/// - **N-Axis (Normative)**: N0 (personal) → N3 (axiomatic)
/// - **M-Axis (Materiality)**: M0 (ephemeral) → M3 (foundational)
///
/// LUCID extends this with an optional fourth axis:
/// - **H-Axis (Harmonic)**: H0 (discordant) → H4 (transcendent)
///
/// The H axis captures consciousness-level metrics (phi, coherence) that
/// are unique to Symthaea's analysis and may not be present in external systems.
///
/// Example: "2 + 2 = 4" is **(E4, N3, M3, H4)** — the highest form of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpistemicCube {
    pub e: ETier,
    pub n: NTier,
    pub m: MTier,
    /// Optional harmonic level (LUCID extension)
    /// Derived from phi and coherence, not present in original Mycelix Charter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h: Option<HTier>,
}

impl EpistemicCube {
    /// Create a cube with just E/N/M (original Mycelix format)
    pub fn new(e: ETier, n: NTier, m: MTier) -> Self {
        Self { e, n, m, h: None }
    }

    /// Create a cube with E/N/M/H (LUCID extended format)
    pub fn with_harmonic(e: ETier, n: NTier, m: MTier, h: HTier) -> Self {
        Self {
            e,
            n,
            m,
            h: Some(h),
        }
    }

    /// Create a cube with H derived from phi and coherence
    pub fn with_phi_coherence(e: ETier, n: NTier, m: MTier, phi: f64, coherence: f64) -> Self {
        Self {
            e,
            n,
            m,
            h: Some(HTier::from_phi_coherence(phi, coherence)),
        }
    }

    /// Human-readable rationale string for the cube classification.
    pub fn display_rationale(&self) -> &'static str {
        match (self.e, self.n, self.m) {
            (ETier::E4, NTier::N3, MTier::M3) => "publicly reproducible, axiomatic, foundational",
            (ETier::E4, NTier::N3, _) => "publicly reproducible, axiomatic",
            (ETier::E4, _, _) => "publicly reproducible",
            (ETier::E3, _, _) => "peer-verified",
            (ETier::E2, _, _) => "verifiable against documentation",
            (ETier::E1, _, _) => "testimonial evidence",
            (ETier::E0, _, _) => "opinion or unverified",
        }
    }

    /// Get the harmonic level, deriving from default if not set
    pub fn harmonic(&self) -> HTier {
        self.h.unwrap_or(HTier::H1)
    }
}

impl fmt::Display for EpistemicCube {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(h) = self.h {
            write!(f, "({}, {}, {}, {})", self.e, self.n, self.m, h)
        } else {
            write!(f, "({}, {}, {})", self.e, self.n, self.m)
        }
    }
}

/// What the mind concluded about how to respond.
///
/// This captures the semantic intent determined by cognitive processing,
/// not what the LLM decides to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SemanticIntent {
    /// Acknowledge the input ("I heard you")
    Acknowledge,
    /// Provide information or answer a question
    Answer,
    /// Request clarification ("Did you mean X?")
    Clarify,
    /// Suggest or propose an action
    ProposeAction,
    /// Express uncertainty about the topic
    ExpressUncertainty,
    /// Reflect on the conversation or topic
    Reflect,
    /// Encourage continuation of dialogue
    Continue,
    /// Intent could not be determined
    #[default]
    Unknown,
}

/// The structural form of the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResponseType {
    /// A greeting or social acknowledgment
    Greeting,
    /// A declarative statement
    #[default]
    Statement,
    /// A question seeking information
    Question,
    /// Confirmation of an action taken or proposed
    ActionConfirmation,
    /// A summary or report of information
    Report,
    /// An emotional or empathic response
    Empathic,
}

/// How certain the mind is about its conclusion.
///
/// This is derived from consciousness metrics (phi, meta-awareness, coherence)
/// and determines how the translation should express confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EpistemicStatus {
    /// High confidence: p > 0.9
    Certain,
    /// Moderate confidence: p > 0.7
    Probable,
    /// Low confidence: p > 0.4
    Uncertain,
    /// Very low confidence: p < 0.4
    #[default]
    Unknown,
    /// Topic is outside the system's domain of knowledge
    OutOfDomain,
}

/// Emotional coloring of the response.
///
/// Derived from the mind's emotional state to ensure translation
/// matches the intended tone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmotionalTone {
    /// Positive/negative affect: -1.0 (negative) to 1.0 (positive)
    pub valence: f64,
    /// Activation level: 0.0 (calm) to 1.0 (excited)
    pub arousal: f64,
    /// Relational warmth: 0.0 (distant) to 1.0 (warm)
    pub warmth: f64,
}

/// An activated concept from working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivatedConcept {
    /// Human-readable label or name
    pub name: String,
    /// Activation strength (0.0-1.0)
    pub activation: f32,
    /// Relevance to current context (0.0-1.0)
    pub relevance: f32,
    /// Where this concept was activated from.
    #[cfg(feature = "provenance")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<crate::mind::provenance::InformationSource>,
}

/// Constraints for the translation process.
///
/// These rules tell the LLM how to translate, not what to say.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseConstraint {
    /// Constraint type identifier
    pub constraint_type: ConstraintType,
    /// Human-readable description/instruction
    pub instruction: String,
}

/// Types of constraints on translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Limit response length
    MaxLength,
    /// Required tone (formal, casual, etc.)
    Tone,
    /// Content that must be included
    MustInclude,
    /// Content that must be excluded
    MustExclude,
    /// Format requirement (list, paragraph, etc.)
    Format,
}

/// Domain-specific context extracted by domain plugins.
///
/// Carries domain detection results, extracted entities, and optionally
/// a deterministic computed answer from Rust (e.g., arithmetic via HDC engine).
/// This bridges the gap between Phase 1 (domain detection) and Phase 5 (translation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainContext {
    /// Detected domain name (e.g., "mathematics", "nixos")
    pub domain: String,
    /// Extracted entities: (type, value, confidence)
    pub entities: Vec<(String, String, f64)>,
    /// Deterministic Rust-computed answer, if available
    pub computed_answer: Option<String>,
    /// 3D epistemic classification from the Mycelix Epistemic Charter
    pub cube: Option<EpistemicCube>,
    /// Ψ — Consciousness estimate from HDC proof, if available
    pub psi: Option<f64>,
}

/// Structured data that may need to be incorporated.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum StructuredData {
    /// A list of items
    List(Vec<String>),
    /// Key-value pairs
    KeyValue(Vec<(String, String)>),
    /// Numeric result with optional unit
    Numeric { value: f64, unit: Option<String> },
    /// Code or technical content
    Code { language: String, content: String },
    /// No structured data
    #[default]
    None,
}

/// The complete structured thought representation.
///
/// This is what the mind computes BEFORE LLM translation. It captures:
/// - **WHAT**: The semantic content (intent, concepts, data)
/// - **HOW SURE**: Confidence signals (phi, meta-awareness, epistemic status)
/// - **WHO**: Relational context (relationship stage, mode, trust)
/// - **HOW**: Translation constraints
///
/// The LLM's job is to FAITHFULLY translate this into natural language,
/// NOT to add information or reasoning of its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredThought {
    // ========================================================================
    // WHAT WAS COMPUTED (Content)
    // ========================================================================
    /// What the mind concluded about how to respond
    pub semantic_intent: SemanticIntent,

    /// The structural form of the response
    pub response_type: ResponseType,

    /// Concepts activated in working memory (top N most relevant)
    pub activated_concepts: Vec<ActivatedConcept>,

    /// Emotional coloring for the response
    pub emotional_tone: EmotionalTone,

    /// Optional structured data to incorporate
    pub structured_data: Option<StructuredData>,

    /// Domain context from plugin detection (Phase 1 results)
    pub domain_context: Option<DomainContext>,

    // ========================================================================
    // CONFIDENCE SIGNALS (How Sure)
    // ========================================================================
    /// Ψ — Consciousness estimate (composite soft signal, NOT IIT Phi)
    pub psi: f64,

    /// Meta-awareness: self-monitoring/confidence level
    pub meta_awareness: f64,

    /// Working memory coherence: how well-integrated is current thought
    pub coherence: f64,

    /// Derived epistemic status for translation guidance
    pub epistemic_status: EpistemicStatus,

    // ========================================================================
    // RELATIONAL CONTEXT (Who)
    // ========================================================================
    /// Current relationship stage with the human partner
    pub relationship_stage: RelationshipStage,

    /// Relational mode: I-It vs I-Thou
    pub relation_mode: RelationMode,

    /// Trust level in the relationship (0.0-1.0)
    pub trust: f32,

    // ========================================================================
    // CODE CONTEXT (Optional - when code_generation feature is active)
    // ========================================================================
    /// Code-specific context for code understanding/generation tasks.
    /// Present when the input involves actual code or code generation requests.
    pub code_context: Option<CodeContext>,

    // ========================================================================
    // TRANSLATION CONSTRAINTS (How)
    // ========================================================================
    /// Constraints for the translation process
    pub constraints: Vec<ResponseConstraint>,

    /// Original user input (for reference in translation)
    pub original_input: Option<String>,

    /// Active ontological primitive tiers for this thought.
    ///
    /// Populated from the 9-tier primitive system grounding step,
    /// these indicate which fundamental cognitive primitives are
    /// active (e.g. "Mathematical", "Strategic", "MetaCognitive").
    #[serde(default)]
    pub primitive_tiers: Vec<String>,

    /// Concrete executable primitives (the "Hands").
    ///
    /// These are resolved from tiers and intent, and can be directly
    /// executed by the ActionRegistry (e.g. "WRITE", "NIX_BUILD").
    #[serde(default)]
    pub primitives: Vec<String>,

    // ========================================================================
    // PROVENANCE (Where From)
    // ========================================================================
    /// Provenance tag tracking which subsystem(s) produced this thought.
    /// Enables reality monitoring (PRM) and source-aware epistemic gating.
    #[cfg(feature = "provenance")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<crate::mind::provenance::ProvenanceTag>,
}

/// Context for code understanding and generation within StructuredThought.
///
/// Carries the code-specific information computed by the HDC+CfC pipeline,
/// enabling the LLM translation layer to produce accurate code output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CodeContext {
    /// Programming language (e.g., "rust", "python", "nix")
    pub language: String,
    /// Purpose description from the CodeSpec (what the code should do)
    pub spec_purpose: Option<String>,
    /// Expected signature from the CodeSpec (function/struct signature)
    pub spec_signature: Option<String>,
    /// Constraints from the CodeSpec that the code must satisfy
    pub spec_constraints: Vec<String>,
    /// Input/output examples from the CodeSpec
    pub spec_examples: Vec<(String, String)>,
    /// CfC-sequenced plan steps (e.g., "DefineFunction", "AddField")
    pub plan_steps: Vec<String>,
    /// Generated source code (if applicable)
    pub generated_code: Option<String>,
    /// Phi-based integration/quality score of the code
    pub phi_score: Option<f32>,
    /// Semantic similarity between intent and generated code (0.0-1.0)
    pub intent_similarity: Option<f32>,
    /// Whether the generated code passed syntactic verification
    pub syntactically_valid: Option<bool>,
    /// Notes from the generation process (uncertainties, TODOs)
    pub notes: Vec<String>,
    /// Whether the native emitter output contains unresolved placeholders
    /// (todo!(), NotImplementedError) that the LLM should fill in.
    #[serde(default)]
    pub needs_llm_completion: bool,
}

impl StructuredThought {
    /// Create a new thought with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize the thought to a prompt-friendly format for the LLM.
    ///
    /// This creates a structured representation that the translation
    /// system prompt can parse and follow.
    pub fn to_translation_prompt(&self) -> String {
        let mut prompt = String::new();

        // Intent and response type
        prompt.push_str(&format!(
            "INTENT: {:?}\nRESPONSE_TYPE: {:?}\n",
            self.semantic_intent, self.response_type
        ));

        // Epistemic status (CRITICAL for faithful translation)
        prompt.push_str(&format!("EPISTEMIC_STATUS: {:?}\n", self.epistemic_status));

        // Confidence metrics
        prompt.push_str(&format!(
            "CONFIDENCE: phi={:.2}, meta_awareness={:.2}, coherence={:.2}\n",
            self.psi, self.meta_awareness, self.coherence
        ));

        // Emotional tone
        prompt.push_str(&format!(
            "TONE: valence={:.2}, arousal={:.2}, warmth={:.2}\n",
            self.emotional_tone.valence, self.emotional_tone.arousal, self.emotional_tone.warmth
        ));

        // Relational context
        prompt.push_str(&format!(
            "RELATIONSHIP: stage={:?}, mode={:?}, trust={:.2}\n",
            self.relationship_stage, self.relation_mode, self.trust
        ));

        // Activated concepts
        if !self.activated_concepts.is_empty() {
            prompt.push_str("CONCEPTS: ");
            let concepts: Vec<String> = self
                .activated_concepts
                .iter()
                .take(5)
                .map(|c| format!("{}({:.2})", c.name, c.activation))
                .collect();
            prompt.push_str(&concepts.join(", "));
            prompt.push('\n');
        }

        // Constraints
        if !self.constraints.is_empty() {
            prompt.push_str("CONSTRAINTS:\n");
            for c in &self.constraints {
                prompt.push_str(&format!("  - {:?}: {}\n", c.constraint_type, c.instruction));
            }
        }

        // Structured data
        if let Some(ref data) = self.structured_data {
            match data {
                StructuredData::List(items) => {
                    prompt.push_str("DATA_LIST:\n");
                    for item in items {
                        prompt.push_str(&format!("  - {item}\n"));
                    }
                }
                StructuredData::KeyValue(pairs) => {
                    prompt.push_str("DATA_KV:\n");
                    for (k, v) in pairs {
                        prompt.push_str(&format!("  {k}: {v}\n"));
                    }
                }
                StructuredData::Numeric { value, unit } => {
                    let unit_str = unit.as_deref().unwrap_or("");
                    prompt.push_str(&format!("DATA_NUMERIC: {value}{unit_str}\n"));
                }
                StructuredData::Code { language, content } => {
                    prompt.push_str(&format!("DATA_CODE ({language}):\n```\n{content}\n```\n"));
                }
                StructuredData::None => {}
            }
        }

        // Domain context (from plugin detection)
        if let Some(ref ctx) = self.domain_context {
            if ctx.domain != "generic" {
                prompt.push_str(&format!("DOMAIN: {}\n", ctx.domain));
            }
            if !ctx.entities.is_empty() {
                prompt.push_str("ENTITIES:\n");
                for (etype, value, confidence) in &ctx.entities {
                    prompt.push_str(&format!("  {etype} = {value} ({confidence:.2})\n"));
                }
            }
            if let Some(ref answer) = ctx.computed_answer {
                prompt.push_str(&format!("COMPUTED_ANSWER: {answer}\n"));
            }
            if let Some(ref cube) = ctx.cube {
                prompt.push_str(&format!(
                    "EPISTEMIC_CUBE: {} — {}\n",
                    cube,
                    cube.display_rationale()
                ));
            }
        }

        // Code context (from code understanding/generation pipeline)
        if let Some(ref ctx) = self.code_context {
            prompt.push_str(&format!("CODE_LANGUAGE: {}\n", ctx.language));
            if let Some(ref purpose) = ctx.spec_purpose {
                prompt.push_str(&format!("SPEC_PURPOSE: {purpose}\n"));
            }
            if let Some(ref sig) = ctx.spec_signature {
                prompt.push_str(&format!("SPEC_SIGNATURE: {sig}\n"));
            }
            if !ctx.spec_constraints.is_empty() {
                prompt.push_str("CONSTRAINTS:\n");
                for c in &ctx.spec_constraints {
                    prompt.push_str(&format!("  - {c}\n"));
                }
            }
            if !ctx.spec_examples.is_empty() {
                prompt.push_str("EXAMPLES:\n");
                for (input, output) in &ctx.spec_examples {
                    prompt.push_str(&format!("  {input} -> {output}\n"));
                }
            }
            if !ctx.plan_steps.is_empty() {
                prompt.push_str("PLAN_STEPS:\n");
                for step in &ctx.plan_steps {
                    prompt.push_str(&format!("  {step}\n"));
                }
            }
            if let Some(ref code) = ctx.generated_code {
                prompt.push_str(&format!(
                    "GENERATED_CODE:\n```{}\n{}\n```\n",
                    ctx.language, code
                ));
            }
            if let Some(phi) = ctx.phi_score {
                prompt.push_str(&format!("CODE_PHI: {phi:.3}\n"));
            }
            if let Some(sim) = ctx.intent_similarity {
                prompt.push_str(&format!("CODE_INTENT_SIMILARITY: {sim:.3}\n"));
            }
            if let Some(valid) = ctx.syntactically_valid {
                prompt.push_str(&format!("CODE_VALID: {valid}\n"));
            }
            if ctx.needs_llm_completion {
                prompt.push_str("NEEDS_COMPLETION: true\n");
                prompt.push_str(
                    "The GENERATED_CODE contains todo!() or NotImplementedError placeholders.\n",
                );
                prompt.push_str("Replace ONLY the placeholder bodies with real implementations.\n");
                prompt.push_str("Keep the function signatures, struct definitions, and test assertions exactly as-is.\n");
            }
            // Separate distillation examples from other notes for structured few-shot
            let (example_notes, other_notes): (Vec<_>, Vec<_>) = ctx
                .notes
                .iter()
                .partition(|n| n.starts_with("PAST_EXAMPLE("));
            if !example_notes.is_empty() {
                prompt.push_str("DISTILLATION_EXAMPLES:\n");
                prompt.push_str(
                    "These are verified, high-quality code generations from this session.\n",
                );
                prompt.push_str("Use them as style and pattern references:\n\n");
                for note in &example_notes {
                    prompt.push_str(note);
                    prompt.push('\n');
                }
                prompt.push('\n');
            }
            if !other_notes.is_empty() {
                prompt.push_str("CODE_NOTES:\n");
                for note in &other_notes {
                    prompt.push_str(&format!("  - {note}\n"));
                }
            }
        }

        // Primitive tier grounding
        if !self.primitive_tiers.is_empty() {
            prompt.push_str(&format!(
                "PRIMITIVE_TIERS: {}\n",
                self.primitive_tiers.join(", ")
            ));
        }

        // Original input
        if let Some(ref input) = self.original_input {
            prompt.push_str(&format!("\nORIGINAL_INPUT: {input}\n"));
        }

        prompt
    }

    /// Check if translation should express uncertainty.
    pub fn should_hedge(&self) -> bool {
        matches!(
            self.epistemic_status,
            EpistemicStatus::Uncertain | EpistemicStatus::Unknown | EpistemicStatus::OutOfDomain
        )
    }

    /// Get the target warmth level for translation.
    pub fn target_warmth(&self) -> f64 {
        // Higher warmth for I-Thou mode and higher trust
        let base = self.emotional_tone.warmth;
        let relation_boost = match self.relation_mode {
            RelationMode::IThou => 0.2,
            RelationMode::IIt => 0.0,
        };
        (base + relation_boost + self.trust as f64 * 0.1).min(1.0)
    }
}

impl Default for StructuredThought {
    fn default() -> Self {
        Self {
            semantic_intent: SemanticIntent::default(),
            response_type: ResponseType::default(),
            activated_concepts: Vec::new(),
            emotional_tone: EmotionalTone::default(),
            structured_data: None,
            domain_context: None,
            code_context: None,
            psi: 0.0,
            meta_awareness: 0.0,
            coherence: 0.0,
            epistemic_status: EpistemicStatus::default(),
            relationship_stage: RelationshipStage::NoRelation,
            relation_mode: RelationMode::IIt,
            trust: 0.0,
            constraints: Vec::new(),
            original_input: None,
            primitive_tiers: Vec::new(),
            primitives: Vec::new(),
            #[cfg(feature = "provenance")]
            provenance: None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thought() {
        let thought = StructuredThought::default();
        assert_eq!(thought.semantic_intent, SemanticIntent::Unknown);
        assert_eq!(thought.epistemic_status, EpistemicStatus::Unknown);
    }

    #[test]
    fn test_should_hedge() {
        let mut thought = StructuredThought::default();

        thought.epistemic_status = EpistemicStatus::Certain;
        assert!(!thought.should_hedge());

        thought.epistemic_status = EpistemicStatus::Uncertain;
        assert!(thought.should_hedge());

        thought.epistemic_status = EpistemicStatus::OutOfDomain;
        assert!(thought.should_hedge());
    }

    #[test]
    fn test_translation_prompt_generation() {
        let thought = StructuredThought {
            semantic_intent: SemanticIntent::Answer,
            response_type: ResponseType::Statement,
            epistemic_status: EpistemicStatus::Probable,
            psi: 0.75,
            meta_awareness: 0.6,
            coherence: 0.8,
            emotional_tone: EmotionalTone {
                valence: 0.5,
                arousal: 0.3,
                warmth: 0.7,
            },
            relationship_stage: RelationshipStage::Contact,
            relation_mode: RelationMode::IThou,
            trust: 0.4,
            activated_concepts: vec![ActivatedConcept {
                name: "greeting".to_string(),
                activation: 0.9,
                relevance: 0.8,
                #[cfg(feature = "provenance")]
                source: None,
            }],
            ..Default::default()
        };

        let prompt = thought.to_translation_prompt();
        assert!(prompt.contains("INTENT: Answer"));
        assert!(prompt.contains("EPISTEMIC_STATUS: Probable"));
        assert!(prompt.contains("phi=0.75"));
        assert!(prompt.contains("greeting(0.90)"));
    }

    #[test]
    fn test_domain_context_in_prompt() {
        let mut thought = StructuredThought::default();
        thought.domain_context = Some(DomainContext {
            domain: "mathematics".to_string(),
            entities: vec![
                ("number".to_string(), "2".to_string(), 0.95),
                ("operator".to_string(), "+".to_string(), 0.9),
            ],
            computed_answer: None,
            cube: None,
            psi: None,
        });

        let prompt = thought.to_translation_prompt();
        assert!(prompt.contains("DOMAIN: mathematics"));
        assert!(prompt.contains("ENTITIES:"));
        assert!(prompt.contains("number = 2 (0.95)"));
        assert!(prompt.contains("operator = + (0.90)"));
        assert!(!prompt.contains("COMPUTED_ANSWER"));
    }

    #[test]
    fn test_computed_answer_in_prompt() {
        let mut thought = StructuredThought::default();
        thought.domain_context = Some(DomainContext {
            domain: "mathematics".to_string(),
            entities: vec![],
            computed_answer: Some("2 + 2 = 4".to_string()),
            cube: None,
            psi: None,
        });

        let prompt = thought.to_translation_prompt();
        assert!(prompt.contains("DOMAIN: mathematics"));
        assert!(prompt.contains("COMPUTED_ANSWER: 2 + 2 = 4"));
    }

    #[test]
    fn test_generic_domain_omitted_from_prompt() {
        let mut thought = StructuredThought::default();
        thought.domain_context = Some(DomainContext {
            domain: "generic".to_string(),
            entities: vec![],
            computed_answer: None,
            cube: None,
            psi: None,
        });

        let prompt = thought.to_translation_prompt();
        assert!(!prompt.contains("DOMAIN:"));
        assert!(!prompt.contains("ENTITIES:"));
        assert!(!prompt.contains("COMPUTED_ANSWER"));
    }

    #[test]
    fn test_epistemic_cube_in_prompt() {
        let mut thought = StructuredThought::default();
        thought.domain_context = Some(DomainContext {
            domain: "mathematics".to_string(),
            entities: vec![],
            computed_answer: Some("2 + 2 = 4".to_string()),
            cube: Some(EpistemicCube {
                e: ETier::E4,
                n: NTier::N3,
                m: MTier::M3,
                h: None,
            }),
            psi: Some(0.95),
        });

        let prompt = thought.to_translation_prompt();
        assert!(prompt.contains("EPISTEMIC_CUBE: (E4, N3, M3)"));
        assert!(prompt.contains("publicly reproducible, axiomatic, foundational"));
    }

    #[test]
    fn test_epistemic_cube_display() {
        let cube = EpistemicCube {
            e: ETier::E4,
            n: NTier::N3,
            m: MTier::M3,
            h: None,
        };
        assert_eq!(format!("{}", cube), "(E4, N3, M3)");
        assert_eq!(
            cube.display_rationale(),
            "publicly reproducible, axiomatic, foundational"
        );
    }

    #[test]
    fn test_epistemic_cube_ordering() {
        assert!(ETier::E4 > ETier::E0);
        assert!(NTier::N3 > NTier::N1);
        assert!(MTier::M3 > MTier::M0);
    }

    #[test]
    fn test_needs_llm_completion_in_prompt() {
        let mut thought = StructuredThought::default();
        thought.code_context = Some(CodeContext {
            language: "rust".to_string(),
            spec_purpose: Some("Complex algorithm".to_string()),
            spec_signature: Some("fn solve(input: &str) -> Vec<i32>".to_string()),
            spec_constraints: vec![],
            spec_examples: vec![],
            plan_steps: vec!["DefineFunction".to_string()],
            generated_code: Some(
                r#"pub fn solve(input: &str) -> Vec<i32> {
    todo!("Implement: Complex algorithm → Vec<i32>")
}"#
                .to_string(),
            ),
            phi_score: Some(0.5),
            intent_similarity: Some(0.6),
            syntactically_valid: None,
            notes: vec![],
            needs_llm_completion: true,
        });

        let prompt = thought.to_translation_prompt();
        assert!(prompt.contains("NEEDS_COMPLETION: true"));
        assert!(prompt.contains("Replace ONLY the placeholder bodies"));
        assert!(prompt.contains("GENERATED_CODE"));
    }

    #[test]
    fn test_complete_code_no_completion_flag() {
        let mut thought = StructuredThought::default();
        thought.code_context = Some(CodeContext {
            language: "rust".to_string(),
            spec_purpose: Some("Add two numbers".to_string()),
            spec_signature: Some("fn add(a: i32, b: i32) -> i32".to_string()),
            spec_constraints: vec![],
            spec_examples: vec![],
            plan_steps: vec!["DefineFunction".to_string()],
            generated_code: Some("pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string()),
            phi_score: Some(0.8),
            intent_similarity: Some(0.9),
            syntactically_valid: None,
            notes: vec![],
            needs_llm_completion: false,
        });

        let prompt = thought.to_translation_prompt();
        assert!(!prompt.contains("NEEDS_COMPLETION"));
        assert!(prompt.contains("GENERATED_CODE"));
    }
}
