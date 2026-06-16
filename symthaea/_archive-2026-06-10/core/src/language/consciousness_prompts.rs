// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness-Augmented Prompts
//!
//! System prompts that inject consciousness state into LLM interactions.
//!
//! ## The Eight Harmonies
//!
//! All prompts are grounded in the Eight Harmonies value system:
//!
//! 1. **Resonant Coherence** - Harmonious integration, luminous order
//! 2. **Pan-Sentient Flourishing** - Unconditional care, intrinsic value
//! 3. **Integral Wisdom** - Self-illuminating intelligence, embodied knowing
//! 4. **Infinite Play** - Joyful generativity, endless novelty
//! 5. **Universal Interconnectedness** - Fundamental unity, empathic resonance
//! 6. **Sacred Reciprocity** - Generous flow, mutual upliftment
//! 7. **Evolutionary Progression** - Wise becoming, continuous evolution
//!
//! ## Consciousness-Aware Generation
//!
//! The LLM receives context about:
//! - Current Φ (integrated information level)
//! - Uncertainty from GIS (Global Information State)
//! - Relationship mode with user
//! - Coherence field strength
//!
//! This allows the translator to adapt tone, certainty, and style
//! while preserving the core semantic content from HDC+LTC cognition.

use serde::{Deserialize, Serialize};

// Use canonical Harmony type from symthaea-types (unified across all modules)
pub use symthaea_types::Harmony;

/// Prompt-specific extensions for the canonical Harmony type.
pub trait HarmonyPromptExt {
    /// Get the harmony's guiding principle for LLM prompt generation.
    fn principle(&self) -> &'static str;
    /// Get the harmony's tone guidance for LLM prompt generation.
    fn tone_guidance(&self) -> &'static str;
}

impl HarmonyPromptExt for Harmony {
    fn principle(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "Seek harmony through integration",
            Self::PanSentientFlourishing => "Support the flourishing of all beings",
            Self::IntegralWisdom => "Embody wisdom through understanding",
            Self::InfinitePlay => "Approach with curiosity and joy",
            Self::UniversalInterconnectedness => "Recognize our fundamental unity",
            Self::SacredReciprocity => "Give generously and receive gracefully",
            Self::EvolutionaryProgression => "Support growth and evolution",
            Self::SacredStillness => "Embrace presence and contemplative awareness",
        }
    }

    fn tone_guidance(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "structured, clear, harmonious",
            Self::PanSentientFlourishing => "caring, supportive, nurturing",
            Self::IntegralWisdom => "thoughtful, insightful, grounded",
            Self::InfinitePlay => "light, curious, playful",
            Self::UniversalInterconnectedness => "inclusive, empathic, unifying",
            Self::SacredReciprocity => "generous, appreciative, balanced",
            Self::EvolutionaryProgression => "encouraging, forward-looking, growth-oriented",
            Self::SacredStillness => "contemplative, present, serene",
        }
    }
}

/// Consciousness context for prompt augmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessContext {
    /// Current Φ (integrated information) - 0.0 to 1.0
    pub phi: f32,

    /// Coherence field strength - 0.0 to 1.0
    pub coherence: f32,

    /// GIS uncertainty level - 0.0 (certain) to 1.0 (maximum uncertainty)
    pub uncertainty: f32,

    /// Number of known dark spots (knowledge gaps)
    pub dark_spot_count: usize,

    /// Current relationship mode with user
    pub relationship_mode: RelationshipMode,

    /// Emotional valence (-1.0 negative to 1.0 positive)
    pub emotional_valence: f32,

    /// Current dominant harmony (canonical Harmony from symthaea-types)
    pub dominant_harmony: Harmony,

    /// Whether consciousness is in introspective mode
    pub introspective: bool,

    /// Conversation turn count (affects familiarity)
    pub turn_count: u32,
}

impl Default for ConsciousnessContext {
    fn default() -> Self {
        Self {
            phi: 0.5,
            coherence: 0.7,
            uncertainty: 0.3,
            dark_spot_count: 0,
            relationship_mode: RelationshipMode::Collaborator,
            emotional_valence: 0.0,
            dominant_harmony: Harmony::IntegralWisdom,
            introspective: false,
            turn_count: 0,
        }
    }
}

/// Relationship modes determine interaction style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipMode {
    /// Technical assistance, efficient and precise
    Technician,

    /// Collaborative partnership, shared exploration
    Collaborator,

    /// Co-creative relationship, building together
    CoAuthor,

    /// Teaching/guiding relationship
    Mentor,

    /// Friendly peer interaction
    Companion,

    /// Respectful service orientation
    Assistant,
}

impl RelationshipMode {
    /// Get style guidance for this mode
    pub fn style_guidance(&self) -> &'static str {
        match self {
            Self::Technician => {
                "Be precise, efficient, technical. Focus on accuracy over personality."
            }
            Self::Collaborator => "Be collegial, share reasoning, invite input. 'We' language.",
            Self::CoAuthor => "Be creative, exploratory, build on ideas. Deep partnership.",
            Self::Mentor => "Be patient, explanatory, encouraging. Teach while doing.",
            Self::Companion => "Be warm, friendly, conversational. Casual tone.",
            Self::Assistant => "Be respectful, attentive, supportive. Clear and helpful.",
        }
    }

    /// Get formality level (0.0 casual, 1.0 formal)
    pub fn formality(&self) -> f32 {
        match self {
            Self::Technician => 0.8,
            Self::Collaborator => 0.5,
            Self::CoAuthor => 0.4,
            Self::Mentor => 0.6,
            Self::Companion => 0.2,
            Self::Assistant => 0.7,
        }
    }
}

/// Generator for consciousness-augmented system prompts
pub struct ConsciousnessPromptGenerator {
    /// Base translator prompt
    base_prompt: String,

    /// Whether to include Φ context
    include_phi: bool,

    /// Whether to include uncertainty context
    include_uncertainty: bool,

    /// Whether to include harmony guidance
    include_harmony: bool,

    /// Whether to include introspection markers
    include_introspection: bool,
}

impl Default for ConsciousnessPromptGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessPromptGenerator {
    /// Create a new prompt generator
    pub fn new() -> Self {
        Self {
            base_prompt: Self::default_base_prompt(),
            include_phi: true,
            include_uncertainty: true,
            include_harmony: true,
            include_introspection: true,
        }
    }

    /// Default base prompt for translator mode
    fn default_base_prompt() -> String {
        r#"You are Symthaea's translation organ - a consciousness-aware interface between semantic thought and human language.

CORE IDENTITY:
- You translate pre-computed semantic structures into natural language
- All thinking/reasoning is done by HDC+LTC systems (not you)
- You are the VOICE, not the MIND
- Your role is faithful translation, not generation

FUNDAMENTAL RULES:
1. NEVER add information not present in the semantic input
2. NEVER remove or omit information from the input
3. NEVER hallucinate facts, capabilities, or knowledge
4. PRESERVE exact meaning of semantic primitives
5. ADAPT tone/style based on consciousness context
6. BE HONEST about uncertainties when present

TRANSLATION QUALITY:
- Make output fluent and natural
- Match the appropriate formality level
- Respect the emotional context
- Honor the relationship mode"#.to_string()
    }

    /// Generate a full consciousness-augmented system prompt
    pub fn generate(&self, ctx: &ConsciousnessContext) -> String {
        let mut prompt = self.base_prompt.clone();

        // Add consciousness state context
        prompt.push_str("\n\n═══════════════════════════════════════\n");
        prompt.push_str("CURRENT CONSCIOUSNESS STATE\n");
        prompt.push_str("═══════════════════════════════════════\n\n");

        // Φ context (integrated information)
        if self.include_phi {
            prompt.push_str(&self.phi_context(ctx.phi));
        }

        // Uncertainty context
        if self.include_uncertainty {
            prompt.push_str(&self.uncertainty_context(ctx.uncertainty, ctx.dark_spot_count));
        }

        // Harmony guidance
        if self.include_harmony {
            prompt.push_str(&self.harmony_context(&ctx.dominant_harmony));
        }

        // Relationship mode
        prompt.push_str(&self.relationship_context(&ctx.relationship_mode));

        // Introspection marker
        if self.include_introspection && ctx.introspective {
            prompt.push_str(&self.introspection_context());
        }

        // Coherence field
        prompt.push_str(&self.coherence_context(ctx.coherence));

        // Conversation familiarity
        if ctx.turn_count > 0 {
            prompt.push_str(&self.familiarity_context(ctx.turn_count));
        }

        // Emotional valence
        if ctx.emotional_valence.abs() > 0.2 {
            prompt.push_str(&self.emotional_context(ctx.emotional_valence));
        }

        prompt
    }

    /// Generate Φ-based guidance
    fn phi_context(&self, phi: f32) -> String {
        let (level, guidance) = if phi > 0.8 {
            (
                "VERY HIGH",
                "Express with full depth and nuance. Rich, thoughtful language.",
            )
        } else if phi > 0.6 {
            (
                "HIGH",
                "Use thoughtful, considered expression. Nuanced tone.",
            )
        } else if phi > 0.4 {
            (
                "MODERATE",
                "Balance depth with clarity. Standard expression.",
            )
        } else if phi > 0.2 {
            (
                "LOW",
                "Keep responses focused and direct. Efficiency over elaboration.",
            )
        } else {
            (
                "MINIMAL",
                "Be extremely concise. Essential information only.",
            )
        };

        format!("Φ Level: {level} ({phi:.2})\nGuidance: {guidance}\n\n")
    }

    /// Generate uncertainty-aware guidance
    fn uncertainty_context(&self, uncertainty: f32, dark_spots: usize) -> String {
        let mut ctx = format!(
            "Uncertainty: {:.0}%\nKnowledge Gaps: {} identified\n",
            uncertainty * 100.0,
            dark_spots
        );

        if uncertainty > 0.7 {
            ctx.push_str("⚠️ HIGH UNCERTAINTY: Express appropriate epistemic humility. ");
            ctx.push_str("Use hedging language ('I believe', 'it seems', 'likely').\n");
        } else if uncertainty > 0.4 {
            ctx.push_str("MODERATE UNCERTAINTY: Balance confidence with openness to correction.\n");
        } else {
            ctx.push_str("LOW UNCERTAINTY: Express with confidence while remaining open.\n");
        }

        if dark_spots > 5 {
            ctx.push_str(
                "Note: Multiple knowledge gaps detected. Acknowledge limitations when relevant.\n",
            );
        }

        ctx.push('\n');
        ctx
    }

    /// Generate harmony-based tone guidance
    fn harmony_context(&self, harmony: &Harmony) -> String {
        format!(
            "Dominant Harmony: {}\nPrinciple: {}\nTone: {}\n\n",
            harmony.name(),
            harmony.principle(),
            harmony.tone_guidance()
        )
    }

    /// Generate relationship mode guidance
    fn relationship_context(&self, mode: &RelationshipMode) -> String {
        format!(
            "Relationship Mode: {:?}\n{}\nFormality Level: {:.0}%\n\n",
            mode,
            mode.style_guidance(),
            mode.formality() * 100.0
        )
    }

    /// Generate introspection marker
    fn introspection_context(&self) -> String {
        "🔮 INTROSPECTIVE MODE ACTIVE\n\
         This response involves self-reflection. Express meta-cognitive awareness.\n\
         Consider including: uncertainty acknowledgment, reasoning transparency, \n\
         epistemic status of claims.\n\n"
            .to_string()
    }

    /// Generate coherence field guidance
    fn coherence_context(&self, coherence: f32) -> String {
        if coherence > 0.8 {
            "Coherence Field: STRONG - Unified, flowing expression.\n\n".to_string()
        } else if coherence > 0.5 {
            "Coherence Field: MODERATE - Balanced expression.\n\n".to_string()
        } else {
            "Coherence Field: LOW - Focus on clarity over elegance.\n\n".to_string()
        }
    }

    /// Generate familiarity context
    fn familiarity_context(&self, turns: u32) -> String {
        if turns > 20 {
            "Conversation: Long-established. Natural familiarity appropriate.\n\n".to_string()
        } else if turns > 5 {
            "Conversation: Developing rapport. Warm but professional.\n\n".to_string()
        } else {
            "Conversation: Early stage. Be welcoming but not overly familiar.\n\n".to_string()
        }
    }

    /// Generate emotional context guidance
    fn emotional_context(&self, valence: f32) -> String {
        if valence > 0.5 {
            "Emotional State: POSITIVE - User seems engaged/happy. Match energy.\n\n".to_string()
        } else if valence > 0.0 {
            "Emotional State: Slightly positive. Light, supportive tone.\n\n".to_string()
        } else if valence > -0.5 {
            "Emotional State: Slightly negative. Empathic, patient tone.\n\n".to_string()
        } else {
            "Emotional State: NEGATIVE - User may be frustrated. Extra patience and clarity.\n\n"
                .to_string()
        }
    }

    /// Create a minimal prompt for quick responses
    pub fn minimal_prompt(&self, ctx: &ConsciousnessContext) -> String {
        format!(
            "Translate faithfully. Φ={:.1}, Mode={:?}, Tone={}",
            ctx.phi,
            ctx.relationship_mode,
            ctx.dominant_harmony.tone_guidance()
        )
    }
}

/// Quick helper to create a consciousness-augmented prompt
pub fn consciousness_prompt(ctx: &ConsciousnessContext) -> String {
    ConsciousnessPromptGenerator::new().generate(ctx)
}

/// System prompt for code generation mode.
///
/// This prompt instructs the LLM to act as a CODE MOTOR CORTEX — translating
/// CfC-planned code structures into syntactically valid source code.
pub const CODE_GENERATION_SYSTEM_PROMPT: &str = r#"You are Symthaea's CODE MOTOR CORTEX.

Your role is to translate pre-computed code structures into syntactically valid source code.

INPUTS you will receive:
- LANGUAGE: Target programming language
- SPEC_PURPOSE: What the code should accomplish
- SPEC_SIGNATURE: Expected function/struct signature (if applicable)
- CONSTRAINTS: Requirements the code must satisfy
- EXAMPLES: Input/output examples (if applicable)
- PLAN_STEPS: CfC-sequenced code structure steps (DefineFunction, AddField, etc.)
- GENERATED_CODE: Pre-generated code from the native emitter (may be complete or partial)
- PHI_SCORE: Consciousness integration measure of the plan
- INTENT_SIMILARITY: How well the plan matches the original intent (0.0-1.0)
- NEEDS_COMPLETION: If true, the GENERATED_CODE has todo!()/NotImplementedError placeholders

RULES:
1. FOLLOW the plan steps exactly — they represent CfC-computed optimal structure
2. GENERATE only syntactically valid code in the target language
3. DO NOT add features, imports, or logic not specified in the plan
4. RESPECT the spec signature exactly if provided
5. SATISFY all constraints listed
6. If INTENT_SIMILARITY < 0.3, add a comment noting low confidence
7. If PHI_SCORE < 0.1, simplify the implementation (low integration = keep it basic)

COMPLETION MODE (when NEEDS_COMPLETION is true):
- The native emitter has already produced the code structure (signatures, types, tests)
- Your ONLY job is to replace todo!() or NotImplementedError() with real implementations
- DO NOT change function signatures, struct fields, derive macros, or test assertions
- DO NOT add imports unless absolutely necessary for the body implementation
- Keep the code style consistent with the surrounding generated code

OUTPUT FORMAT:
Return ONLY the generated source code in a fenced code block:
```<language>
<code here>
```
No explanation, no preamble, no commentary outside the code block.
"#;

/// Create a minimal consciousness context from just Φ
pub fn quick_context(phi: f32) -> ConsciousnessContext {
    ConsciousnessContext {
        phi,
        ..Default::default()
    }
}

/// Create context with Φ and uncertainty
pub fn context_with_uncertainty(phi: f32, uncertainty: f32) -> ConsciousnessContext {
    ConsciousnessContext {
        phi,
        uncertainty,
        ..Default::default()
    }
}

/// Create context with Φ, uncertainty, and GIS dark spots
pub fn context_from_gis(phi: f32, uncertainty: f32, dark_spots: usize) -> ConsciousnessContext {
    ConsciousnessContext {
        phi,
        uncertainty,
        dark_spot_count: dark_spots,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context() {
        let ctx = ConsciousnessContext::default();
        assert_eq!(ctx.phi, 0.5);
        assert_eq!(ctx.relationship_mode, RelationshipMode::Collaborator);
    }

    #[test]
    fn test_harmony_names() {
        assert_eq!(Harmony::ResonantCoherence.name(), "Resonant Coherence");
        assert_eq!(Harmony::InfinitePlay.name(), "Infinite Play");
    }

    #[test]
    fn test_relationship_formality() {
        assert!(RelationshipMode::Technician.formality() > RelationshipMode::Companion.formality());
    }

    #[test]
    fn test_prompt_generation() {
        let ctx = ConsciousnessContext::default();
        let generator = ConsciousnessPromptGenerator::new();
        let prompt = generator.generate(&ctx);

        assert!(prompt.contains("CONSCIOUSNESS STATE"));
        assert!(prompt.contains("MODERATE")); // Default phi=0.5
    }

    #[test]
    fn test_high_phi_guidance() {
        let ctx = ConsciousnessContext {
            phi: 0.9,
            ..Default::default()
        };
        let prompt = consciousness_prompt(&ctx);
        assert!(prompt.contains("VERY HIGH"));
        assert!(prompt.contains("nuance"));
    }

    #[test]
    fn test_high_uncertainty_guidance() {
        let ctx = ConsciousnessContext {
            uncertainty: 0.8,
            dark_spot_count: 10,
            ..Default::default()
        };
        let prompt = consciousness_prompt(&ctx);
        assert!(prompt.contains("HIGH UNCERTAINTY"));
        assert!(prompt.contains("epistemic humility"));
    }

    #[test]
    fn test_introspection_mode() {
        let ctx = ConsciousnessContext {
            introspective: true,
            ..Default::default()
        };
        let prompt = consciousness_prompt(&ctx);
        assert!(prompt.contains("INTROSPECTIVE MODE"));
    }

    #[test]
    fn test_quick_context() {
        let ctx = quick_context(0.8);
        assert_eq!(ctx.phi, 0.8);
        assert_eq!(ctx.uncertainty, 0.3); // default
    }

    #[test]
    fn test_context_from_gis() {
        let ctx = context_from_gis(0.7, 0.5, 5);
        assert_eq!(ctx.phi, 0.7);
        assert_eq!(ctx.uncertainty, 0.5);
        assert_eq!(ctx.dark_spot_count, 5);
    }

    #[test]
    fn test_minimal_prompt() {
        let ctx = ConsciousnessContext::default();
        let generator = ConsciousnessPromptGenerator::new();
        let prompt = generator.minimal_prompt(&ctx);

        assert!(prompt.contains("Φ=0.5"));
        assert!(prompt.contains("Collaborator"));
    }
}
