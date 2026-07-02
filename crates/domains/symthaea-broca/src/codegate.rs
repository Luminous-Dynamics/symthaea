// codegate.rs
// Real CodeGate implementation for Symthaea Broca
// Registers the full LanguageGateRegistry (all 13+ IaC + general languages)
// Calls apply_with_emotion_and_language() for consciousness-gated emission
// Integrates with ThoughtChannels for intent, valence, arousal, epistemic scores

use crate::emotional_gating_integration::modulate_by_emotion;
use crate::encoder::ThoughtChannels; // Assumed: has language_intent(), prompt_contains_any(), valence(), arousal(), moral_score() etc.
use crate::gating::EpistemicCubeGate;
use crate::language_gates::LanguageGateRegistry; // for emotional modulation
use crate::tokenizer::BpeTokenizer;

pub struct CodeGate {
    language_gate_registry: LanguageGateRegistry,
    epistemic_cube_gate: EpistemicCubeGate,
    tokenizer: BpeTokenizer,
    base_temperature: f32,
    base_top_p: f32,
    current_temperature: f32,
    current_top_p: f32,
}

impl CodeGate {
    pub fn new(tokenizer: &crate::tokenizer::BpeTokenizer) -> Self {
        // assume tokenizer exists
        let language_gate_registry = LanguageGateRegistry::new(tokenizer);
        let epistemic_cube_gate = EpistemicCubeGate::new(tokenizer);
        println!(
            "✅ CodeGate initialized with {} language/IaC gates registered",
            language_gate_registry.list_gates().len()
        );
        CodeGate {
            language_gate_registry,
            epistemic_cube_gate,
            tokenizer: tokenizer.clone(),
            base_temperature: 0.85,
            base_top_p: 0.88,
            current_temperature: 0.85,
            current_top_p: 0.88,
        }
    }

    /// Main apply: detects intent via registry, applies emotional modulation, boosts logits
    pub fn apply(&mut self, logits: &mut [f32], channels: &ThoughtChannels) {
        self.apply_with_emotion_and_language(logits, channels);
    }

    /// Full gated apply: language + emotion + (future epistemic)
    pub fn apply_with_emotion_and_language(
        &mut self,
        logits: &mut [f32],
        channels: &ThoughtChannels,
    ) {
        // 1. Detect language/intent using full registry (now includes OpenTofu, CDK, Bicep, Crossplane, Argo CD, Helm)
        let language_gate = self.language_gate_registry.detect_intent(channels);
        let base_gate_strength = if let Some(gate) = language_gate {
            println!(
                "🔍 Detected IaC/Language gate: {} (base boost: {:.1})",
                gate.name, gate.base_boost
            );
            gate.base_boost
        } else {
            1.8 // default Rust/general
        };

        // 2. Modulate by emotional state (frustration -> creative/explore, positive -> precise)
        let (temperature, top_p, final_gate_strength) = modulate_by_emotion(
            channels,
            self.base_temperature,
            self.base_top_p,
            base_gate_strength,
        );

        // 3. Store for downstream generate()
        self.current_temperature = temperature;
        self.current_top_p = top_p;

        // 4. Apply the (emotionally modulated) language gate boost to logits
        if let Some(gate) = language_gate {
            self.language_gate_registry
                .apply_gate(logits, gate, final_gate_strength);
            println!(
                "🚀 Applied {} gate boost (final strength: {:.2}) | temp={:.2} top_p={:.2}",
                gate.name, final_gate_strength, temperature, top_p
            );
        }

        // 5. Apply epistemic-certainty-based hallucination suppression on top of the
        //    language/emotion gating above (penalizes speculative identifier-like
        //    tokens when the E-axis signals low certainty).
        self.epistemic_cube_gate
            .apply_strict_code_gate(logits, channels, &self.tokenizer);
    }

    /// Visualization helper: ASCII bar chart of gate boosts (for debugging generation)
    pub fn visualize_gate_boosts(&self, active_intent: Option<&str>) {
        println!("\n📊 Gate Boost Visualization (current emotional modulation applied):");
        let gates = self.language_gate_registry.list_gates();
        let max_boost = gates.iter().map(|g| g.base_boost).fold(0.0, f32::max);
        for gate in gates {
            let bar_len = ((gate.base_boost / max_boost) * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            let marker = if active_intent.map_or(false, |i| {
                gate.name.to_lowercase().contains(&i.to_lowercase())
                    || i.to_lowercase().contains(&gate.name.to_lowercase())
            }) {
                " ← ACTIVE"
            } else {
                ""
            };
            println!(
                "  {:<12} | {:<40} {:.1}{}",
                gate.name, bar, gate.base_boost, marker
            );
        }
        println!();
    }

    pub fn get_current_sampling_params(&self) -> (f32, f32) {
        (self.current_temperature, self.current_top_p)
    }
}

// Example usage in generation loop (e.g. in broca_generate or nix_repair equivalent)
pub fn example_codegate_flow() {
    // In real: let tokenizer = BpeTokenizer::new(...);
    // let mut codegate = CodeGate::new(&tokenizer);
    // let mut channels = ThoughtChannels::default();
    // channels.set_language_intent("opentofu".to_string());
    // channels.set_valence(-0.3); channels.set_arousal(0.8); // frustrated -> explore
    // let mut logits = vec![0.0f32; 50257]; // vocab size example
    // codegate.apply(&mut logits, &channels);
    // codegate.visualize_gate_boosts(Some("opentofu"));
    println!("Example CodeGate flow ready for integration with Broca's main generation loop.");
}
