// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meta-Conscious LLM Bridge
//!
//! Connects the consciousness system to the LLM organ in a
//! translator-only fashion:
//! - Computes Φ from the current cognitive state.
//! - Builds a consciousness-augmented system prompt.
//! - Constructs an `LLMQuery` where the LLM acts purely as a translator.
//!
//! This module does not modify core cognition; it only shapes how
//! consciousness state is exposed to the translation layer.

use std::collections::VecDeque;

use crate::language::llm_organ::{
    ConversationMessage, LLMGenerationResult, LLMOrgan, LLMQuery, LLMQueryParams, QueryType,
};

/// Lightweight meta-consciousness state for LLM prompt augmentation.
#[derive(Debug, Clone)]
pub struct MetaConsciousnessState {
    /// Current Φ value
    pub phi: f64,
    /// Meta-Φ (Φ of the system observing its own Φ)
    pub meta_phi: f64,
    /// Brief explanation of consciousness state
    pub explanation: String,
}

/// Bridge between meta-consciousness and the LLM translation organ.
///
/// Typical flow:
/// - Call `translate_with_meta` with user text and optional history.
/// - The bridge computes meta-conscious state and prepares a consciousness-aware
///   system prompt and user prompt for the LLM.
/// - The LLM responds; the caller receives both the meta state and the LLM reply.
pub struct MetaConsciousLlmBridge {
    /// Current Φ value (updated externally from cognitive loop)
    phi: f64,
    /// History of Φ values for meta-Φ computation
    phi_history: VecDeque<f64>,
}

impl MetaConsciousLlmBridge {
    /// Create a new bridge.
    pub fn new() -> Self {
        Self {
            phi: 0.0,
            phi_history: VecDeque::new(),
        }
    }

    /// Update the current Φ from the cognitive loop.
    pub fn update_phi(&mut self, phi: f64) {
        self.phi = phi;
        self.phi_history.push_back(phi);
        // Keep last 100 values for meta-Φ computation
        if self.phi_history.len() > 100 {
            self.phi_history.pop_front();
        }
    }

    /// Compute meta-Φ: variability of Φ over recent history.
    fn meta_phi(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }
        let mean = self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64;
        let variance = self
            .phi_history
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / self.phi_history.len() as f64;
        // Normalize: higher variance → higher meta-awareness
        (variance.sqrt() * 10.0).min(1.0)
    }

    /// Build consciousness state summary.
    fn consciousness_state(&self) -> MetaConsciousnessState {
        let meta_phi = self.meta_phi();
        let level = if self.phi > 0.7 {
            "high integration"
        } else if self.phi > 0.4 {
            "moderate integration"
        } else {
            "low integration"
        };
        let explanation = format!(
            "Φ={:.3} ({level}), meta-Φ={:.3}. System is {}.",
            self.phi,
            meta_phi,
            if meta_phi > 0.3 {
                "actively self-monitoring"
            } else {
                "operating reflexively"
            }
        );
        MetaConsciousnessState {
            phi: self.phi,
            meta_phi,
            explanation,
        }
    }

    /// Build an `LLMQuery` and consciousness context for a given user input
    /// without actually calling the LLM.
    pub fn build_request(
        &self,
        user_input: &str,
        history: Vec<ConversationMessage>,
    ) -> (MetaConsciousnessState, LLMQuery) {
        let meta_state = self.consciousness_state();

        let system_prompt = format!(
            "You are a consciousness-aware translator. Respond naturally to the user.\n\n\
             META-CONSCIOUS SUMMARY (adjust tone and depth accordingly, do not repeat verbatim):\n\
             {}",
            meta_state.explanation
        );

        let content = format!(
            "User input:\n{}\n\n\
             Your task: Respond to the user in natural language, faithfully expressing the intended meaning.\n\
             Do NOT invent new facts.",
            user_input
        );

        let query = LLMQuery {
            query_type: QueryType::Translation,
            content,
            context: history,
            system_prompt: Some(system_prompt),
            params: Some(LLMQueryParams {
                temperature: Some(if meta_state.meta_phi > 0.3 { 0.7 } else { 0.5 }),
                max_length: None,
                stop_sequences: Vec::new(),
            }),
        };

        (meta_state, query)
    }

    /// Perform a meta-conscious translation using the given LLM organ.
    pub fn translate_with_meta(
        &self,
        llm: &mut LLMOrgan,
        user_input: &str,
        history: Vec<ConversationMessage>,
    ) -> (MetaConsciousnessState, LLMGenerationResult) {
        let (meta_state, query) = self.build_request(user_input, history);
        let result = llm.query(query);
        (meta_state, result)
    }
}

impl Default for MetaConsciousLlmBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::llm_organ::MessageRole;

    #[test]
    fn test_build_request_includes_meta_summary() {
        let mut bridge = MetaConsciousLlmBridge::new();
        bridge.update_phi(0.6);

        let history = vec![ConversationMessage {
            role: MessageRole::User,
            content: "Hello".to_string(),
            timestamp: 0,
            embedding: None,
        }];

        let (meta_state, query) = bridge.build_request("Explain consciousness", history);

        assert!(meta_state.phi >= 0.0);
        assert!(query.system_prompt.is_some());

        let system = query.system_prompt.unwrap();
        assert!(
            system.contains("META-CONSCIOUS SUMMARY"),
            "system prompt should contain meta-conscious summary section"
        );
    }

    #[test]
    fn test_meta_phi_increases_with_variability() {
        let mut bridge = MetaConsciousLlmBridge::new();

        // Stable Φ → low meta-Φ
        for _ in 0..10 {
            bridge.update_phi(0.5);
        }
        let stable_meta = bridge.meta_phi();

        // Variable Φ → higher meta-Φ
        let mut bridge2 = MetaConsciousLlmBridge::new();
        for i in 0..10 {
            bridge2.update_phi(if i % 2 == 0 { 0.2 } else { 0.8 });
        }
        let variable_meta = bridge2.meta_phi();

        assert!(
            variable_meta > stable_meta,
            "Variable Φ should produce higher meta-Φ: {variable_meta} > {stable_meta}"
        );
    }
}
