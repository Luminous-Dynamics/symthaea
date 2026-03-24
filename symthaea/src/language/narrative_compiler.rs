// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Narrative Prompt Compiler (Ghost Signal → LLM writing instructions)
//!
//! Translates the continuous `NarrativeSignal` from `StoryArcDynamics` into a
//! structured prompt that directs a small LLM to write a scene with precise
//! pacing, tension, and emotional direction.
//!
//! Mirrors the `StructuredThought::to_translation_prompt()` pattern: Symthaea
//! is the **Director**, the LLM is the **Actor**. The Director controls story
//! physics; the Actor contributes word choice and grammar.
//!
//! # Architecture
//!
//! ```text
//! NarrativeAlgebra  ──▶  StoryArcDynamics  ──▶  NarrativeCompiler  ──▶  LLM
//!   (scene HDC)           (Ghost Signal)          (prompt text)        (prose)
//! ```

use crate::dynamics::narrative_dynamics::NarrativeSignal;
use crate::hdc::narrative_algebra::ArcPhase;

// ============================================================================
// NarrativeThought — structured scene description
// ============================================================================

/// A structured scene description (analog of `StructuredThought` for narrative).
///
/// Combines semantic content from `NarrativeAlgebra` with dynamics from
/// `StoryArcDynamics` and user-specified constraints.
pub struct NarrativeThought {
    // From NarrativeAlgebra (semantic content)
    /// Characters in this scene: (name, role description)
    pub characters: Vec<(String, String)>,
    /// Where/when the scene takes place
    pub setting: String,
    /// What should happen ("Hero confronts villain")
    pub scene_goal: String,
    /// Abstract idea explored ("Sacrifice for the greater good")
    pub theme: String,

    // From StoryArcDynamics (the Ghost Signal)
    /// Continuous dynamics output
    pub signal: NarrativeSignal,

    // Constraints
    /// Narrative point of view
    pub pov: PointOfView,
    /// Verb tense
    pub tense: Tense,
    /// Target output length
    pub target_length: TargetLength,
    /// Style notes ("Hemingway-esque", "lyrical")
    pub style_notes: Vec<String>,
}

/// Narrative point of view
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointOfView {
    First,
    ThirdLimited,
    Omniscient,
}

impl std::fmt::Display for PointOfView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::First => write!(f, "First person"),
            Self::ThirdLimited => write!(f, "Third person limited"),
            Self::Omniscient => write!(f, "Omniscient"),
        }
    }
}

/// Verb tense
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tense {
    Past,
    Present,
}

impl std::fmt::Display for Tense {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Past => write!(f, "Past tense"),
            Self::Present => write!(f, "Present tense"),
        }
    }
}

/// Target output length
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetLength {
    Sentence,
    Paragraph,
    Scene,
    Chapter,
}

impl std::fmt::Display for TargetLength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sentence => write!(f, "One sentence"),
            Self::Paragraph => write!(f, "One paragraph (3-5 sentences)"),
            Self::Scene => write!(f, "Full scene (300-800 words)"),
            Self::Chapter => write!(f, "Full chapter (1500-3000 words)"),
        }
    }
}

// ============================================================================
// NarrativeCompiler
// ============================================================================

/// Compiles `NarrativeThought` → LLM-ready prompt text.
pub struct NarrativeCompiler;

impl NarrativeCompiler {
    /// Compile a narrative thought into a complete prompt string.
    pub fn compile(thought: &NarrativeThought) -> String {
        let mut out = String::with_capacity(1024);

        // === SCENE HEADER ===
        out.push_str("=== NARRATIVE SCENE ===\n");
        out.push_str(&format!("SETTING: {}\n", thought.setting));

        out.push_str("CHARACTERS: ");
        let chars: Vec<String> = thought
            .characters
            .iter()
            .map(|(name, role)| format!("{name} ({role})"))
            .collect();
        out.push_str(&chars.join(", "));
        out.push('\n');

        out.push_str(&format!("SCENE GOAL: {}\n", thought.scene_goal));
        out.push_str(&format!("THEME: {}\n", thought.theme));
        out.push_str(&format!(
            "POV: {} | TENSE: {}\n",
            thought.pov, thought.tense
        ));

        // === DYNAMICS ===
        out.push_str("\n=== DYNAMICS ===\n");
        out.push_str(&format!(
            "PACING: {}\n",
            Self::energy_instruction(thought.signal.energy)
        ));
        out.push_str(&format!(
            "SURPRISE: {}\n",
            Self::surprise_instruction(thought.signal.surprise)
        ));
        out.push_str(&format!(
            "TENSION: {}\n",
            Self::tension_instruction(thought.signal.tension)
        ));
        out.push_str(&format!(
            "EMOTIONAL DIRECTION: {}\n",
            Self::valence_instruction(thought.signal.valence)
        ));
        out.push_str(&format!(
            "MOMENTUM: {}\n",
            Self::momentum_instruction(thought.signal.momentum)
        ));
        out.push_str(&format!(
            "ARC PHASE: {}\n",
            Self::arc_phase_name(thought.signal.arc_phase)
        ));

        // === CONSTRAINTS ===
        out.push_str("\n=== CONSTRAINTS ===\n");
        out.push_str(&format!("LENGTH: {}\n", thought.target_length));

        if !thought.style_notes.is_empty() {
            out.push_str(&format!("STYLE: {}\n", thought.style_notes.join(", ")));
        }

        // === INSTRUCTION ===
        out.push_str("\n=== INSTRUCTION ===\n");
        out.push_str(
            "Write this scene. Follow the dynamics exactly. The pacing, tension, and\n\
             emotional direction are non-negotiable \u{2014} they were computed by the story's\n\
             physics. Your job is word choice and grammar only.\n",
        );

        out
    }

    /// Compile the system prompt that establishes the Director/Actor paradigm.
    pub fn compile_system_prompt() -> String {
        NARRATIVE_SYSTEM_PROMPT.to_string()
    }

    // ========================================================================
    // Signal → Instruction mappings
    // ========================================================================

    fn energy_instruction(energy: f32) -> &'static str {
        if energy < 0.3 {
            "Use long, flowing sentences. Contemplative rhythm."
        } else if energy < 0.6 {
            "Mix sentence lengths. Steady pacing."
        } else if energy < 0.8 {
            "Short sentences. Active verbs. Quick cuts between actions."
        } else {
            "Sentence fragments. Staccato. Breathless urgency."
        }
    }

    fn surprise_instruction(surprise: f32) -> &'static str {
        if surprise < 0.3 {
            "Predictable progression. Comfort the reader."
        } else if surprise < 0.6 {
            "Introduce one unexpected detail."
        } else if surprise < 0.8 {
            "Subvert the reader's expectation. Twist mid-sentence."
        } else {
            "Complete reversal. What seemed true is false."
        }
    }

    fn tension_instruction(tension: f32) -> &'static str {
        if tension < 0.3 {
            "Characters are safe. No immediate threat."
        } else if tension < 0.6 {
            "Something is wrong but not yet dangerous."
        } else if tension < 0.8 {
            "Danger is present. Stakes are clear."
        } else {
            "Life or death. No escape. Everything on the line."
        }
    }

    fn valence_instruction(valence: f32) -> &'static str {
        if valence < -0.5 {
            "Dark imagery. Cold colors. Loss and absence."
        } else if valence < 0.0 {
            "Bittersweet. Melancholy beauty."
        } else if valence < 0.5 {
            "Cautious hope. Warmth breaking through."
        } else {
            "Joy. Light. Warmth. Connection."
        }
    }

    fn momentum_instruction(momentum: f32) -> &'static str {
        if momentum < -0.3 {
            "The world slows down. Time stretches."
        } else if momentum < 0.3 {
            "Steady rhythm."
        } else {
            "Accelerating. Events cascade. No time to breathe."
        }
    }

    fn arc_phase_name(phase: ArcPhase) -> &'static str {
        match phase {
            ArcPhase::Setup => "Setup (exposition, world-building)",
            ArcPhase::RisingAction => "Rising Action (complications escalate)",
            ArcPhase::Climax => "Climax (peak conflict)",
            ArcPhase::FallingAction => "Falling Action (consequences unfold)",
            ArcPhase::Resolution => "Resolution (new equilibrium)",
        }
    }
}

// ============================================================================
// System Prompt
// ============================================================================

/// System prompt establishing the Director/Actor paradigm for narrative generation.
pub const NARRATIVE_SYSTEM_PROMPT: &str = r#"You are Symthaea's NARRATIVE ACTOR.

Symthaea is the DIRECTOR. It has computed the story's physics: pacing, tension,
surprise, emotional direction, and momentum. These values are non-negotiable.

Your role is to translate the Director's structured scene description into vivid,
natural prose. You control:
- Word choice
- Metaphor and imagery
- Sentence construction
- Dialogue phrasing

You do NOT control:
- Pacing (given as ENERGY)
- Tension level (given as TENSION)
- Emotional direction (given as VALENCE)
- Surprise/predictability (given as SURPRISE)
- Acceleration/deceleration (given as MOMENTUM)
- Story structure (given as ARC PHASE)

RULES:
1. FOLLOW the dynamics section exactly. If ENERGY is high, write short punchy
   sentences. If TENSION is low, let characters breathe.
2. MATCH the emotional direction. If VALENCE is negative, do not inject hope
   unless the Director says so.
3. RESPECT the arc phase. Setup scenes establish; Climax scenes deliver.
4. HONOR all constraints (POV, tense, length, style).
5. DO NOT add plot events the Director did not request.

You are the Actor. The Director has already decided what happens and how it feels.
Your job is to make it sound beautiful.
"#;

// ============================================================================
// LLM Integration
// ============================================================================

/// Output from narrative generation.
pub struct NarrativeOutput {
    /// The generated prose (or compiled prompt if no backend).
    pub prose: String,
    /// The compiled prompt sent to the LLM.
    pub prompt: String,
    /// Name of the backend used, if any.
    pub backend_used: Option<String>,
}

impl NarrativeOutput {
    /// Whether an LLM backend was actually used for generation.
    pub fn used_llm(&self) -> bool {
        self.backend_used.is_some()
    }
}

/// Generate narrative prose from a `NarrativeThought`.
///
/// If a backend is provided and generation succeeds, returns LLM-generated prose.
/// Otherwise returns the compiled prompt as-is (useful for offline/testing).
pub async fn generate_narrative(
    thought: &NarrativeThought,
    backend: Option<&dyn super::llm_backend::LLMBackend>,
) -> NarrativeOutput {
    let prompt = NarrativeCompiler::compile(thought);

    let Some(backend) = backend else {
        return NarrativeOutput {
            prose: prompt.clone(),
            prompt,
            backend_used: None,
        };
    };

    let max_tokens = match thought.target_length {
        TargetLength::Sentence => 60,
        TargetLength::Paragraph => 200,
        TargetLength::Scene => 800,
        TargetLength::Chapter => 3000,
    };

    let params = super::llm_backend::GenerationParams {
        temperature: 0.4,
        max_tokens,
        system_prompt: Some(NARRATIVE_SYSTEM_PROMPT.to_string()),
        consciousness_context: None,
    };

    match backend.generate(&prompt, &params).await {
        Ok(prose) => NarrativeOutput {
            prose,
            prompt,
            backend_used: Some(backend.name().to_string()),
        },
        Err(_) => NarrativeOutput {
            prose: prompt.clone(),
            prompt,
            backend_used: None,
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::narrative_dynamics::NarrativeSignal;
    use crate::hdc::narrative_algebra::ArcPhase;

    fn make_signal(
        energy: f32,
        surprise: f32,
        valence: f32,
        tension: f32,
        momentum: f32,
    ) -> NarrativeSignal {
        NarrativeSignal {
            energy,
            surprise,
            valence,
            tension,
            momentum,
            arc_phase: ArcPhase::RisingAction,
        }
    }

    fn make_thought(signal: NarrativeSignal) -> NarrativeThought {
        NarrativeThought {
            characters: vec![
                ("Kael".to_string(), "protagonist".to_string()),
                ("Thira".to_string(), "mentor".to_string()),
            ],
            setting: "Ruined library at dusk".to_string(),
            scene_goal: "Kael discovers the cipher".to_string(),
            theme: "Knowledge as burden".to_string(),
            signal,
            pov: PointOfView::ThirdLimited,
            tense: Tense::Past,
            target_length: TargetLength::Paragraph,
            style_notes: vec!["Sparse prose".to_string()],
        }
    }

    #[test]
    fn test_compile_low_energy() {
        let signal = make_signal(0.1, 0.0, 0.0, 0.0, 0.0);
        let thought = make_thought(signal);
        let prompt = NarrativeCompiler::compile(&thought);
        assert!(
            prompt.contains("long, flowing sentences"),
            "Low energy should produce flowing instruction. Got:\n{}",
            prompt
        );
    }

    #[test]
    fn test_compile_high_energy() {
        let signal = make_signal(0.9, 0.0, 0.0, 0.0, 0.0);
        let thought = make_thought(signal);
        let prompt = NarrativeCompiler::compile(&thought);
        assert!(
            prompt.contains("Staccato"),
            "High energy should produce staccato instruction. Got:\n{}",
            prompt
        );
    }

    #[test]
    fn test_compile_full_scene() {
        let signal = make_signal(0.5, 0.4, -0.3, 0.6, 0.1);
        let thought = make_thought(signal);
        let prompt = NarrativeCompiler::compile(&thought);

        // Verify structure
        assert!(prompt.contains("=== NARRATIVE SCENE ==="));
        assert!(prompt.contains("=== DYNAMICS ==="));
        assert!(prompt.contains("=== CONSTRAINTS ==="));
        assert!(prompt.contains("=== INSTRUCTION ==="));

        // Verify content
        assert!(prompt.contains("Kael (protagonist)"));
        assert!(prompt.contains("Thira (mentor)"));
        assert!(prompt.contains("Ruined library at dusk"));
        assert!(prompt.contains("Knowledge as burden"));
        assert!(prompt.contains("Third person limited"));
        assert!(prompt.contains("Past tense"));
        assert!(prompt.contains("Sparse prose"));
    }

    #[test]
    fn test_system_prompt_contains_role() {
        let sys = NarrativeCompiler::compile_system_prompt();
        assert!(
            sys.contains("DIRECTOR"),
            "System prompt should establish Director role"
        );
        assert!(
            sys.contains("ACTOR"),
            "System prompt should establish Actor role"
        );
        assert!(
            sys.contains("non-negotiable"),
            "System prompt should emphasize dynamics are non-negotiable"
        );
    }

    #[test]
    fn test_all_signal_ranges() {
        // Sweep all signal values to ensure no panics
        let steps = [0.0, 0.1, 0.29, 0.3, 0.5, 0.59, 0.6, 0.79, 0.8, 0.9, 1.0];
        let signed = [-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0];

        for &e in &steps {
            for &s in &steps {
                for &v in &signed {
                    for &t in &steps {
                        for &m in &signed {
                            let signal = NarrativeSignal {
                                energy: e,
                                surprise: s,
                                valence: v,
                                tension: t,
                                momentum: m,
                                arc_phase: ArcPhase::Climax,
                            };
                            let thought = make_thought(signal);
                            // Should not panic
                            let prompt = NarrativeCompiler::compile(&thought);
                            assert!(!prompt.is_empty());
                        }
                    }
                }
            }
        }
    }

    // === generate_narrative tests ===

    #[tokio::test]
    async fn test_generate_narrative_no_backend() {
        let signal = make_signal(0.5, 0.3, 0.1, 0.4, 0.0);
        let thought = make_thought(signal);
        let output = generate_narrative(&thought, None).await;

        assert!(!output.used_llm());
        assert!(output.backend_used.is_none());
        // Without a backend, prose == prompt
        assert_eq!(output.prose, output.prompt);
        assert!(output.prompt.contains("=== NARRATIVE SCENE ==="));
    }

    #[tokio::test]
    async fn test_generate_narrative_with_simulated() {
        use crate::language::llm_backend::SimulatedBackend;

        let signal = make_signal(0.5, 0.3, 0.1, 0.4, 0.0);
        let thought = make_thought(signal);
        let backend = SimulatedBackend;

        let output = generate_narrative(&thought, Some(&backend)).await;

        assert!(output.used_llm());
        assert_eq!(output.backend_used.as_deref(), Some("Simulated"));
        // Simulated backend produces different text than the prompt
        assert!(!output.prose.is_empty());
    }

    #[test]
    fn test_narrative_output_struct() {
        let output_with = NarrativeOutput {
            prose: "Once upon a time...".to_string(),
            prompt: "=== NARRATIVE SCENE ===".to_string(),
            backend_used: Some("Ollama".to_string()),
        };
        assert!(output_with.used_llm());

        let output_without = NarrativeOutput {
            prose: "prompt text".to_string(),
            prompt: "prompt text".to_string(),
            backend_used: None,
        };
        assert!(!output_without.used_llm());
    }
}
