//! Meta-Conscious LLM Bridge
//!
//! Connects the meta-consciousness engine to the LLM organ in a
//! translator-only fashion:
//! - Uses `MetaConversationCore` to compute Φ and meta-Φ from user text.
//! - Builds a consciousness-augmented system prompt via `ConsciousnessPromptGenerator`.
//! - Constructs an `LlmRequest` where the LLM acts purely as a translator.
//!
//! This module does not modify core cognition; it only shapes how
//! consciousness state is exposed to the translation layer.

use anyhow::Result;

use crate::hdc::meta_conscious_conversation::MetaConversationCore;
use crate::language::consciousness_prompts::{
    ConsciousnessContext,
    ConsciousnessPromptGenerator,
    SevenHarmonies,
    RelationshipMode,
};
use crate::language::llm_organ::{LlmOrgan, LlmRequest, LlmResponse, Message, Role};

/// Bridge between meta-consciousness and the LLM translation organ.
///
/// Typical flow:
/// - Call `translate_with_meta` with user text and optional history.
/// - The bridge computes meta-conscious state and prepares a consciousness-aware
///   system prompt and user prompt for the LLM.
/// - The LLM responds; the caller receives both the meta state and the LLM reply.
pub struct MetaConsciousLlmBridge {
    core: MetaConversationCore,
    prompt_generator: ConsciousnessPromptGenerator,
}

impl MetaConsciousLlmBridge {
    /// Create a new bridge with default configs and the given number
    /// of HV16 components for meta-reflection.
    pub fn new(num_components: usize) -> Result<Self> {
        Ok(Self {
            core: MetaConversationCore::new(num_components)?,
            prompt_generator: ConsciousnessPromptGenerator::new(),
        })
    }

    /// Access the underlying meta-conversation core.
    pub fn core(&self) -> &MetaConversationCore {
        &self.core
    }

    /// Mutable access to the underlying meta-conversation core.
    pub fn core_mut(&mut self) -> &mut MetaConversationCore {
        &mut self.core
    }

    /// Build an `LlmRequest` and consciousness context for a given user input
    /// without actually calling the LLM.
    ///
    /// This is useful for testing and for callers that want to inspect or modify
    /// the request before sending.
    pub fn build_request(
        &mut self,
        user_input: &str,
        history: Vec<Message>,
    ) -> Result<(crate::hdc::meta_consciousness::MetaConsciousnessState, LlmRequest, ConsciousnessContext)> {
        let meta_state = self.core.reflect_on_text(user_input)?;

        // Map Φ and meta-Φ into a ConsciousnessContext for prompt augmentation.
        let phi = meta_state.phi as f32;
        let meta_phi = meta_state.meta_phi as f32;
        let introspective = meta_phi > 0.3;

        // For now, we do not have a rich emotional model here; keep neutral.
        let mut ctx = ConsciousnessContext::default();
        ctx.phi = phi.clamp(0.0, 1.0);
        ctx.introspective = introspective;
        ctx.dominant_harmony = SevenHarmonies::IntegralWisdom;
        ctx.relationship_mode = RelationshipMode::Collaborator;
        ctx.turn_count = history.len() as u32;

        let system_prompt = {
            let mut base = self.prompt_generator.generate(&ctx);
            base.push_str("\n\nMETA-CONSCIOUS SUMMARY (for the translator, not to be repeated verbatim):\n");
            base.push_str(&meta_state.explanation);
            base
        };

        // User-facing prompt: instruct the LLM to act strictly as translator.
        let prompt = format!(
            "User input:\n{}\n\n\
             Your task: Respond to the user in natural language, faithfully expressing the intended meaning.\n\
             Do NOT invent new facts. Use the meta-conscious summary from the system prompt only to\n\
             adjust tone, depth, and style, not to add content.\n",
            user_input
        );

        let request = LlmRequest {
            prompt,
            system: Some(system_prompt),
            history,
            context_embedding: None,
            expected_domain: None,
            temperature_override: None,
        };

        Ok((meta_state, request, ctx))
    }

    /// Perform a meta-conscious translation using the given LLM organ.
    ///
    /// - Computes meta-conscious state for `user_input`.
    /// - Builds a consciousness-augmented `LlmRequest`.
    /// - Calls `llm.generate(request, φ)` and returns both meta state and response.
    pub async fn translate_with_meta(
        &mut self,
        llm: &mut LlmOrgan,
        user_input: &str,
        history: Vec<Message>,
    ) -> Result<(crate::hdc::meta_consciousness::MetaConsciousnessState, LlmResponse)> {
        let (meta_state, request, _ctx) = self.build_request(user_input, history)?;
        let phi = meta_state.phi as f32;

        let response = llm.generate(request, phi).await?;
        Ok((meta_state, response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_request_includes_meta_summary_and_phi() {
        let mut bridge = MetaConsciousLlmBridge::new(4).expect("bridge should initialize");
        let history = vec![
            Message {
                role: Role::User,
                content: "Hello".to_string(),
            },
        ];

        let (meta_state, request, ctx) = bridge
            .build_request("Explain meta-consciousness in simple terms.", history)
            .expect("build_request should succeed");

        assert!(meta_state.phi >= 0.0);
        assert!(request.system.is_some());

        let system = request.system.unwrap();
        assert!(
            system.contains("META-CONSCIOUS SUMMARY"),
            "system prompt should contain meta-conscious summary section"
        );
        assert!(
            system.contains("CONSCIOUSNESS STATE") || system.contains("Φ Level"),
            "system prompt should contain consciousness state context"
        );

        assert!(
            ctx.phi >= 0.0 && ctx.phi <= 1.0,
            "consciousness context φ should be clamped to [0,1]"
        );
    }
}

