// emotional_gating_integration.rs
// Emotional-Epistemic Coupling + Frustration Loop for Symthaea Broca
// Wires compiler verification failures into ThoughtChannels (valence/arousal)
// Modulates generation parameters (temperature, top_p, gate strength) in real time

use crate::compiler_trainer::CompilerVerdict;
use crate::encoder::ThoughtChannels;
use crate::gating::CodeGate;
use crate::language_gates::LanguageGateRegistry;

/// Call this after every compiler verification step (in nix_repair.rs or code_orchestrator.rs)
pub fn apply_frustration_trigger(
    channels: &mut ThoughtChannels,
    verdict: &CompilerVerdict,
    consecutive_failures: usize,
) {
    if let CompilerVerdict::Fail { .. } = verdict {
        // Frustration accumulator (0.0 = calm, 1.0 = highly frustrated)
        // Since we don't have error_count in the real verdict, use a default value
        let error_count = 1;
        let frustration = ((error_count as f32 + consecutive_failures as f32) * 0.12).min(1.0);

        // Update valence (emotional "mood" - negative = bad)
        let current_valence = channels.valence();
        let new_valence = (current_valence * 0.65 - frustration * 0.55).clamp(-1.0, 1.0);
        channels.set_valence(new_valence);

        // Update arousal (energy level - high = ready to explore or panic)
        let current_arousal = channels.arousal();
        let new_arousal = (current_arousal * 0.55 + frustration * 0.85).min(1.0);
        channels.set_arousal(new_arousal);

        // Optional: log for training signal / introspection
        if frustration > 0.55 {
            tracing::warn!(
                "HIGH FRUSTRATION: {} consecutive failures (valence={:.2}, arousal={:.2})",
                consecutive_failures,
                new_valence,
                new_arousal
            );
        }
    } else {
        // Success → gently reduce arousal and boost valence
        let current_valence = channels.valence();
        channels.set_valence((current_valence * 0.85 + 0.15).min(1.0));

        let current_arousal = channels.arousal();
        channels.set_arousal(current_arousal * 0.75); // calm down after success
    }
}

/// Modulates generation parameters based on current emotional state
/// Call this inside CodeGate::apply() or the main generation loop
pub fn modulate_by_emotion(
    channels: &ThoughtChannels,
    base_temperature: f32,
    base_top_p: f32,
    language_gate_strength: f32, // from LanguageGateRegistry
) -> (f32, f32, f32) {
    let arousal = channels.arousal();
    let valence = channels.valence();

    let mut temperature = base_temperature;
    let mut top_p = base_top_p;
    let mut gate_strength = language_gate_strength;

    // === FRUSTRATED STATE (high arousal + low/negative valence) ===
    // → More creative / exploratory mode (like a human engineer "stepping back")
    if arousal > 0.62 && valence < 0.05 {
        temperature = (base_temperature * 1.38).min(1.35);
        top_p = 0.93;
        gate_strength *= 0.55; // Relax language gates → allow more diverse syntax
    // This helps escape local minima when stuck on one language's patterns
    }
    // === FOCUSED + POSITIVE STATE (high arousal + positive valence) ===
    // → Precise, high-quality exploitation
    else if arousal > 0.72 && valence > 0.25 {
        temperature = base_temperature * 0.82;
        top_p = 0.76;
        gate_strength *= 1.25; // Stronger language adherence
    }
    // === LOW ENERGY (low arousal) ===
    // → More conservative / deterministic
    else if arousal < 0.35 {
        temperature = base_temperature * 0.75;
        top_p = 0.68;
    }

    // Clamp values to sane ranges
    temperature = temperature.clamp(0.5, 1.4);
    top_p = top_p.clamp(0.6, 0.98);
    gate_strength = gate_strength.clamp(0.3, 3.5);

    (temperature, top_p, gate_strength)
}

/// Example integration point inside CodeGate (add this method or modify apply())
impl CodeGate {
    pub fn apply_with_emotion_and_language(
        &mut self,
        logits: &mut [f32],
        channels: &ThoughtChannels,
        language_registry: &LanguageGateRegistry,
    ) {
        // 1. Detect language and apply base gate
        let language_gate = language_registry.detect_intent(channels);
        let base_gate_strength = if let Some(gate) = language_gate {
            gate.base_boost
        } else {
            1.8
        };

        // 2. Modulate everything by emotional state
        let (temperature, top_p, final_gate_strength) = modulate_by_emotion(
            channels,
            self.base_temperature,
            self.base_top_p,
            base_gate_strength,
        );

        // 3. Apply emotional modulation to sampling params (store for generate())
        self.current_temperature = temperature;
        self.current_top_p = top_p;

        // 4. Apply language gate with emotional modulation
        if let Some(gate) = language_gate {
            language_registry.apply_gate(logits, gate, final_gate_strength);
        }

        // 5. (Optional) Still run existing v5 code-channel scoring + EpistemicCubeGate
        // self.apply_v5_scoring(logits, channels);
        // self.epistemic_cube_gate.apply(...);
    }
}
