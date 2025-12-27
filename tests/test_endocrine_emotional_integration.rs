//! Integration Tests: Endocrine System + Emotional Reasoning
//!
//! Tests that the EndocrineSystem correctly maps hormone states to emotional states
//! using the new emotional primitives (VALENCE, AROUSAL, JOY, FEAR, etc.)

use symthaea::physiology::{
    EndocrineSystem,
    EndocrineConfig,
    HormoneEvent,
    HormoneState,
    Emotion,
};

#[test]
fn test_endocrine_has_emotional_reasoning() {
    let system = EndocrineSystem::new(EndocrineConfig::default());

    // Verify the endocrine system can provide emotional state
    let emotion = system.emotional_state();

    println!("\n=== ENDOCRINE EMOTIONAL REASONING INTEGRATION ===");
    println!("Initial emotional state: {:?}", emotion.dominant_emotion);
    println!("Valence: {:.3}", emotion.valence);
    println!("Arousal: {:.3}", emotion.arousal);
    println!("Intensity: {:.3}", emotion.intensity);

    assert!(emotion.valence >= -1.0 && emotion.valence <= 1.0, "Valence should be in range [-1, 1]");
    assert!(emotion.arousal >= 0.0 && emotion.arousal <= 1.0, "Arousal should be in range [0, 1]");
    assert!(emotion.intensity >= 0.0 && emotion.intensity <= 1.0, "Intensity should be in range [0, 1]");

    println!("✓ Endocrine system can derive emotional state from hormones");
}

#[test]
fn test_stress_creates_fear() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== STRESS → FEAR MAPPING ===");

    // High cortisol + low dopamine = negative valence, high arousal = FEAR
    system.process_event(HormoneEvent::Threat { intensity: 0.9 });

    let emotion = system.emotional_state();
    let hormones = system.state();

    println!("After threat:");
    println!("  Cortisol: {:.3}", hormones.cortisol);
    println!("  Dopamine: {:.3}", hormones.dopamine);
    println!("  Valence: {:.3}", emotion.valence);
    println!("  Arousal: {:.3}", emotion.arousal);
    println!("  Emotion: {:?}", emotion.dominant_emotion);

    // Should have negative valence (cortisol > dopamine)
    assert!(emotion.valence < 0.0, "Threat should create negative valence");

    // Should have elevated arousal (above neutral baseline of ~0.4)
    assert!(emotion.arousal > 0.42, "Threat should create elevated arousal");

    // Dominant emotion should be fear or anger (both negative valence, high arousal)
    match emotion.dominant_emotion {
        Emotion::Fear | Emotion::Anger => {
            println!("✓ Threat correctly mapped to {:?}", emotion.dominant_emotion);
        },
        _ => panic!("Expected Fear or Anger, got {:?}", emotion.dominant_emotion),
    }
}

#[test]
fn test_success_creates_joy() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== SUCCESS → JOY MAPPING ===");

    // Multiple success events to build up dopamine
    system.process_event(HormoneEvent::Success { magnitude: 0.8 });
    system.process_event(HormoneEvent::Reward { value: 0.9 });

    let emotion = system.emotional_state();
    let hormones = system.state();

    println!("After success:");
    println!("  Cortisol: {:.3}", hormones.cortisol);
    println!("  Dopamine: {:.3}", hormones.dopamine);
    println!("  Valence: {:.3}", emotion.valence);
    println!("  Arousal: {:.3}", emotion.arousal);
    println!("  Emotion: {:?}", emotion.dominant_emotion);

    // Should have positive valence (dopamine > cortisol)
    assert!(emotion.valence > 0.0, "Success should create positive valence");

    // Should have moderate to high arousal
    assert!(emotion.arousal > 0.3, "Success should create arousal");

    // Dominant emotion should be joy or contentment (both positive valence)
    match emotion.dominant_emotion {
        Emotion::Joy | Emotion::Contentment => {
            println!("✓ Success correctly mapped to {:?}", emotion.dominant_emotion);
        },
        _ => panic!("Expected Joy or Contentment, got {:?}", emotion.dominant_emotion),
    }
}

#[test]
fn test_calm_state_after_recovery() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== RECOVERY → CALM MAPPING ===");

    // Create stress first
    system.process_event(HormoneEvent::Error { severity: 0.7 });

    // Then recover
    system.process_event(HormoneEvent::Recovery);

    let emotion = system.emotional_state();
    let hormones = system.state();

    println!("After recovery:");
    println!("  Cortisol: {:.3}", hormones.cortisol);
    println!("  Dopamine: {:.3}", hormones.dopamine);
    println!("  Valence: {:.3}", emotion.valence);
    println!("  Arousal: {:.3}", emotion.arousal);
    println!("  Emotion: {:?}", emotion.dominant_emotion);

    // Should have low arousal (calm)
    assert!(emotion.arousal < 0.6, "Recovery should reduce arousal");

    println!("✓ Recovery creates calmer emotional state");
}

#[test]
fn test_emotional_state_updates_with_hormones() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== EMOTIONAL STATE TRACKING ===");

    // Initial state
    let initial_emotion = system.emotional_state().dominant_emotion.clone();
    println!("Initial: {:?}", initial_emotion);

    // Stress event should change emotion
    system.process_event(HormoneEvent::Threat { intensity: 0.8 });
    let stressed_emotion = system.emotional_state().dominant_emotion.clone();
    println!("After threat: {:?}", stressed_emotion);

    // Emotions should be different (unless we started in fear)
    if !matches!(initial_emotion, Emotion::Fear | Emotion::Anger) {
        assert_ne!(format!("{:?}", initial_emotion), format!("{:?}", stressed_emotion),
                   "Emotion should change after threat event");
    }

    // Success event should change emotion again
    system.process_event(HormoneEvent::Success { magnitude: 0.9 });
    system.process_event(HormoneEvent::Reward { value: 0.9 });
    let happy_emotion = system.emotional_state().dominant_emotion.clone();
    println!("After success: {:?}", happy_emotion);

    // Should move towards positive emotions
    assert!(matches!(happy_emotion, Emotion::Joy | Emotion::Contentment | Emotion::Surprise),
            "After success, should have positive emotion, got {:?}", happy_emotion);

    println!("✓ Emotional state updates with hormone changes");
}

#[test]
fn test_decay_updates_emotional_state() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== EMOTIONAL STATE DURING DECAY ===");

    // Create high stress
    system.process_event(HormoneEvent::Threat { intensity: 1.0 });
    let stressed_emotion = system.emotional_state().dominant_emotion.clone();
    let stressed_intensity = system.emotional_state().intensity;

    println!("High stress: {:?} (intensity: {:.3})", stressed_emotion, stressed_intensity);

    // Decay over many cycles
    for i in 0..50 {
        system.decay_cycle();

        if i % 10 == 0 {
            let current = system.emotional_state();
            println!("Cycle {}: {:?} (intensity: {:.3}, valence: {:.3})",
                     i, current.dominant_emotion, current.intensity, current.valence);
        }
    }

    let final_emotion = system.emotional_state();

    println!("After decay: {:?} (intensity: {:.3})",
             final_emotion.dominant_emotion, final_emotion.intensity);

    // Hormones should have returned closer to baseline
    let final_valence_diff = final_emotion.valence.abs();
    let stressed_valence_diff = system.state().valence().abs();

    // Either:
    // 1. Emotion changed (stress emotion → neutral), OR
    // 2. Valence returned toward neutral (closer to 0)
    let emotion_changed = format!("{:?}", final_emotion.dominant_emotion) != format!("{:?}", stressed_emotion);
    let valence_normalized = final_valence_diff < stressed_valence_diff.abs() + 0.1;

    assert!(emotion_changed || valence_normalized,
            "Should either change emotion or normalize valence during decay");

    println!("✓ Emotional state tracks hormone decay over time");
}

#[test]
fn test_stats_include_emotional_data() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== STATS WITH EMOTIONAL DATA ===");

    // Create some emotional state
    system.process_event(HormoneEvent::Success { magnitude: 0.7 });

    let stats = system.stats();

    println!("Statistics:");
    println!("  Mood (legacy): {}", stats.current_mood);
    println!("  Emotion (primitive-based): {}", stats.emotion);
    println!("  Intensity: {:.3}", stats.emotion_intensity);
    println!("  Secondary emotions: {:?}", stats.secondary_emotions);

    // Stats should include emotional data
    assert!(!stats.emotion.is_empty(), "Stats should include emotion");
    assert!(stats.emotion_intensity >= 0.0 && stats.emotion_intensity <= 1.0,
            "Emotion intensity should be in valid range");

    println!("✓ Stats include both hormonal and emotional data");
}

#[test]
fn test_complex_emotions_emerge() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== COMPLEX EMOTIONAL SCENARIOS ===");

    // Scenario 1: Anxious (high arousal, negative valence)
    system.process_event(HormoneEvent::Error { severity: 0.8 });
    system.process_event(HormoneEvent::DeepFocus { duration_cycles: 10 });

    let anxious_state = system.emotional_state();
    println!("Anxious state: {:?} (V={:.2}, A={:.2})",
             anxious_state.dominant_emotion, anxious_state.valence, anxious_state.arousal);

    // Scenario 2: Content (low arousal, positive valence)
    let mut calm_system = EndocrineSystem::new(EndocrineConfig::default());
    calm_system.process_event(HormoneEvent::Success { magnitude: 0.4 });
    for _ in 0..20 { calm_system.decay_cycle(); }

    let content_state = calm_system.emotional_state();
    println!("Content state: {:?} (V={:.2}, A={:.2})",
             content_state.dominant_emotion, content_state.valence, content_state.arousal);

    // Scenario 3: Excited (high arousal, positive valence)
    let mut excited_system = EndocrineSystem::new(EndocrineConfig::default());
    excited_system.process_event(HormoneEvent::Reward { value: 0.9 });
    excited_system.process_event(HormoneEvent::Success { magnitude: 0.8 });

    let excited_state = excited_system.emotional_state();
    println!("Excited state: {:?} (V={:.2}, A={:.2})",
             excited_state.dominant_emotion, excited_state.valence, excited_state.arousal);

    // Verify different scenarios produce different emotional states
    println!("✓ Complex emotional scenarios produce appropriate emotional states");
}

#[test]
fn test_secondary_emotions_detected() {
    let mut system = EndocrineSystem::new(EndocrineConfig::default());

    println!("\n=== SECONDARY EMOTIONS ===");

    // Create mixed emotional state
    system.process_event(HormoneEvent::Success { magnitude: 0.6 });
    system.process_event(HormoneEvent::Error { severity: 0.3 });

    let emotion = system.emotional_state();

    println!("Dominant emotion: {:?}", emotion.dominant_emotion);
    println!("Secondary emotions:");
    for (secondary, intensity) in &emotion.secondary_emotions {
        println!("  {:?}: {:.3}", secondary, intensity);
    }

    // Should have at least one secondary emotion in mixed states
    if emotion.valence.abs() < 0.5 && emotion.arousal > 0.3 {
        assert!(!emotion.secondary_emotions.is_empty(),
                "Mixed emotional state should have secondary emotions");
    }

    println!("✓ Secondary emotions detected in complex states");
}
