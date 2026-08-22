const LLM_ORGAN_SOURCE: &str = include_str!("../../../../src/language/llm_organ.rs");

#[test]
fn ab_harness_snapshot_tracks_current_legacy_runtime_builder() {
    for required in [
        "fn build_translation_prompt(",
        "=== STRUCTURED THOUGHT TO TRANSLATE ===",
        "MOOD_TEMPERATURE: {:.2}",
        "thought.to_translation_prompt()",
        "honest expression of uncertainty",
        "Include hedging language to express uncertainty",
        "it might be",
        "thought.target_warmth()",
        "Maintain a warm, friendly tone",
        "Maintain a neutral, professional tone",
        "Respond ONLY with the translated natural language",
    ] {
        assert!(
            LLM_ORGAN_SOURCE.contains(required),
            "legacy runtime prompt builder changed; update the A/B mirror for: {required}"
        );
    }
}

#[test]
fn ab_harness_snapshot_tracks_current_legacy_control_risks() {
    for required in [
        "FOLLOW all constraints",
        "DO NOT suggest possibilities",
        "guaranteed correct",
    ] {
        assert!(
            LLM_ORGAN_SOURCE.contains(required),
            "legacy translation control changed; update the A/B safety comparison for: {required}"
        );
    }
}
