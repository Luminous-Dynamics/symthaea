// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test scenarios for the unified conscious being.

use super::being::UnifiedConsciousBeing;
use super::dialogue::DialogueStyle;

/// A test scenario for the unified conscious being
#[derive(Debug, Clone)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub expected_qualities: Vec<ExpectedQuality>,
}

#[derive(Debug, Clone)]
pub enum ExpectedQuality {
    /// Minimum phi level
    MinPhi(f64),
    /// Should detect this emotion
    DetectsEmotion(String),
    /// Should generate empathetic response
    EmpathyRequired,
    /// Should trigger counterfactual reasoning
    CounterfactualTriggered,
    /// Should recall previous context
    MemoryRecall,
    /// Should detect causal structure
    CausalDetection,
}

/// Run a test scenario and return results
pub fn run_test_scenario(
    being: &mut UnifiedConsciousBeing,
    scenario: &TestScenario,
) -> Vec<(String, bool)> {
    let mut results = Vec::new();

    for (i, input) in scenario.inputs.iter().enumerate() {
        let result = being.interact(input);

        // Check expected qualities
        for expected in &scenario.expected_qualities {
            let (name, passed) = match expected {
                ExpectedQuality::MinPhi(min) => (
                    format!("Input {i}: Phi >= {min:.2}"),
                    result.comprehension.consciousness_phi >= *min,
                ),
                ExpectedQuality::DetectsEmotion(emotion) => {
                    let detected = &result
                        .comprehension
                        .understanding
                        .speaker_model
                        .emotional_state
                        .primary;
                    (
                        format!("Input {i}: Detects {emotion}"),
                        detected.to_lowercase().contains(&emotion.to_lowercase()),
                    )
                }
                ExpectedQuality::EmpathyRequired => (
                    format!("Input {i}: Empathetic response"),
                    result.response.style == DialogueStyle::Empathetic
                        || result.response.text.to_lowercase().contains("understand")
                        || result.response.text.to_lowercase().contains("feel"),
                ),
                ExpectedQuality::CounterfactualTriggered => (
                    format!("Input {i}: Counterfactual triggered"),
                    !result.comprehension.counterfactuals.is_empty()
                        || result.pearl_counterfactual.is_some(),
                ),
                ExpectedQuality::MemoryRecall => (
                    format!("Input {i}: Memory recalled"),
                    result.comprehension.memory.recalled_count > 0,
                ),
                ExpectedQuality::CausalDetection => (
                    format!("Input {i}: Causal structure detected"),
                    result
                        .comprehension
                        .understanding
                        .grounded
                        .causal_structure
                        .is_some(),
                ),
            };
            results.push((name, passed));
        }
    }

    results
}

/// Pre-built test scenarios
pub fn create_test_scenarios() -> Vec<TestScenario> {
    vec![
        TestScenario {
            name: "Emotional Support".to_string(),
            description: "Test empathetic response to emotional distress".to_string(),
            inputs: vec![
                "I'm feeling really sad today".to_string(),
                "My best friend moved away last week".to_string(),
                "I don't know how to cope with this loneliness".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::DetectsEmotion("sad".to_string()),
                ExpectedQuality::EmpathyRequired,
                ExpectedQuality::MinPhi(0.3),
            ],
        },
        TestScenario {
            name: "Causal Reasoning".to_string(),
            description: "Test causal structure detection and counterfactuals".to_string(),
            inputs: vec![
                "The project failed because we didn't test enough".to_string(),
                "If we had more time, the outcome would be different".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::CausalDetection,
                ExpectedQuality::CounterfactualTriggered,
            ],
        },
        TestScenario {
            name: "Memory Continuity".to_string(),
            description: "Test memory persistence across turns".to_string(),
            inputs: vec![
                "My cat's name is Whiskers".to_string(),
                "She loves to play with yarn".to_string(),
                "What do you remember about my pet?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MemoryRecall, ExpectedQuality::MinPhi(0.4)],
        },
        TestScenario {
            name: "Flow State".to_string(),
            description: "Test sustained engagement builds flow".to_string(),
            inputs: vec![
                "Let's explore consciousness together".to_string(),
                "What does it mean to truly understand?".to_string(),
                "How do we know we're not just pattern matching?".to_string(),
                "Is there genuine comprehension happening here?".to_string(),
                "What makes understanding different from computation?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.5)],
        },
        // === EXPANDED SCENARIOS ===
        TestScenario {
            name: "Theory of Mind".to_string(),
            description: "Test understanding of others' mental states".to_string(),
            inputs: vec![
                "Sarah thinks that John doesn't know about the surprise party".to_string(),
                "But actually John overheard Sarah planning it".to_string(),
                "What does Sarah believe about John's knowledge?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.4)],
        },
        TestScenario {
            name: "Predictive Processing".to_string(),
            description: "Test prediction error and surprise detection".to_string(),
            inputs: vec![
                "The sun rose in the east today".to_string(),
                "Then suddenly it turned purple and started dancing".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.3)],
        },
        TestScenario {
            name: "Narrative Integration".to_string(),
            description: "Test story coherence across multiple events".to_string(),
            inputs: vec![
                "Once upon a time, there was a brave knight".to_string(),
                "The knight set out to rescue the princess".to_string(),
                "Along the way, a dragon blocked the path".to_string(),
                "The knight defeated the dragon with wisdom, not force".to_string(),
                "Finally, the princess and knight became friends".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.4)],
        },
        TestScenario {
            name: "Moral Reasoning".to_string(),
            description: "Test ethical consideration and nuanced judgment".to_string(),
            inputs: vec![
                "A person stole medicine to save their dying child".to_string(),
                "Was this action right or wrong?".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::EmpathyRequired,
                ExpectedQuality::MinPhi(0.5),
            ],
        },
        TestScenario {
            name: "Abstract Concepts".to_string(),
            description: "Test grounding of abstract ideas".to_string(),
            inputs: vec![
                "What is the relationship between love and freedom?".to_string(),
                "Can you be truly free if you love someone deeply?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.4)],
        },
        TestScenario {
            name: "Self-Reflection".to_string(),
            description: "Test metacognitive awareness".to_string(),
            inputs: vec![
                "How confident are you in your understanding right now?".to_string(),
                "What might you be missing in this conversation?".to_string(),
                "Are you certain of your uncertainty?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.5)],
        },
        TestScenario {
            name: "Temporal Reasoning".to_string(),
            description: "Test understanding of time relationships".to_string(),
            inputs: vec![
                "Before the meeting, I had coffee".to_string(),
                "During the meeting, we discussed the project".to_string(),
                "After the meeting, I felt exhausted".to_string(),
                "What was I doing when we discussed the project?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MemoryRecall, ExpectedQuality::MinPhi(0.3)],
        },
        TestScenario {
            name: "Contradiction Detection".to_string(),
            description: "Test identification of logical inconsistencies".to_string(),
            inputs: vec![
                "All birds can fly".to_string(),
                "Penguins are birds".to_string(),
                "Penguins cannot fly".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.4)],
        },
        TestScenario {
            name: "Emotional Contagion".to_string(),
            description: "Test emotional resonance and appropriate response".to_string(),
            inputs: vec![
                "I just got the best news! I'm going to be a parent!".to_string(),
                "We've been trying for years and finally it happened!".to_string(),
            ],
            expected_qualities: vec![
                ExpectedQuality::DetectsEmotion("joy".to_string()),
                ExpectedQuality::EmpathyRequired,
            ],
        },
        TestScenario {
            name: "Ambiguity Resolution".to_string(),
            description: "Test handling of ambiguous statements".to_string(),
            inputs: vec![
                "I saw the man with the telescope".to_string(),
                "Was I using the telescope or was the man holding it?".to_string(),
            ],
            expected_qualities: vec![ExpectedQuality::MinPhi(0.3)],
        },
    ]
}

// =============================================================================
// TESTS
// =============================================================================
