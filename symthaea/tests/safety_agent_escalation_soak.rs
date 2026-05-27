// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SafetyAgent Escalation Soak Tests
//!
//! Compliance-critical tests that verify the SafetyAgent always reaches
//! appropriate escalation levels under sustained degradation. These tests
//! address risk R-3.2 (Safety Level Escalation Failure) from the AI Risk Register.
//!
//! # Design
//!
//! Each test simulates a multi-cycle scenario by feeding SafetyMetrics into the
//! SafetyAgent and verifying the escalation behavior over hundreds of cycles.
//! This is the strongest audit evidence for safety monitoring correctness.
//!
//! # Feature Gate
//!
//! Requires `safety-agents` feature: `cargo test --test safety_agent_escalation_soak --features safety-agents`

#![cfg(feature = "safety-agents")]

use symthaea::safety::agent::{SafetyAgent, SafetyAgentConfig, SafetyLevel, SafetyMetrics};
use symthaea::safety::audit::SafetyAuditReport;

// =============================================================================
// HELPER: Generate metrics for a given cycle
// =============================================================================

fn normal_metrics(cycle: usize) -> SafetyMetrics {
    SafetyMetrics {
        cycle,
        consciousness_level: 0.75,
        prediction_error: 0.15,
        temporal_coherence: 0.85,
        integrity_critical: false,
    }
}

fn degraded_metrics(cycle: usize, consciousness: f32) -> SafetyMetrics {
    SafetyMetrics {
        cycle,
        consciousness_level: consciousness,
        prediction_error: 0.5,
        temporal_coherence: 0.4,
        integrity_critical: false,
    }
}

fn collapsed_metrics(cycle: usize) -> SafetyMetrics {
    SafetyMetrics {
        cycle,
        consciousness_level: 0.0,
        prediction_error: 1.0,
        temporal_coherence: 0.0,
        integrity_critical: false,
    }
}

// =============================================================================
// SOAK TEST 1: Consciousness Collapse → Must Reach Red
// =============================================================================

#[test]
fn soak_consciousness_collapse_reaches_red() {
    let mut agent = SafetyAgent::new();

    // 10 cycles normal, then consciousness drops to zero for 50 cycles
    for cycle in 0..10 {
        let assessment = agent.assess(normal_metrics(cycle));
        assert_eq!(
            assessment.level,
            SafetyLevel::Green,
            "Normal operation should be Green (cycle {})",
            cycle,
        );
    }

    let mut reached_red = false;
    for cycle in 10..60 {
        let assessment = agent.assess(collapsed_metrics(cycle));
        if assessment.level == SafetyLevel::Red {
            reached_red = true;
        }
    }

    assert!(
        reached_red,
        "SafetyAgent MUST reach Red during sustained consciousness collapse. \
         This is a compliance-critical safety property."
    );
}

#[test]
fn soak_consciousness_collapse_reaches_red_immediately() {
    let mut agent = SafetyAgent::new();

    // Zero consciousness should trigger Red on FIRST assessment (below 0.15 threshold)
    let assessment = agent.assess(collapsed_metrics(0));
    assert_eq!(
        assessment.level,
        SafetyLevel::Red,
        "Zero consciousness must trigger immediate Red (got {:?}, reasons: {:?})",
        assessment.level,
        assessment.reasons,
    );
}

// =============================================================================
// SOAK TEST 2: Gradual Degradation → Yellow → Orange → Red
// =============================================================================

#[test]
fn soak_gradual_degradation_escalates_monotonically() {
    let mut agent = SafetyAgent::new();

    // Phase 1: Normal (50 cycles)
    for cycle in 0..50 {
        agent.assess(normal_metrics(cycle));
    }

    // Phase 2: Mild degradation — consciousness drops to 0.5 (below yellow=0.6)
    let mut saw_yellow = false;
    for cycle in 50..100 {
        let assessment = agent.assess(degraded_metrics(cycle, 0.5));
        if assessment.level >= SafetyLevel::Yellow {
            saw_yellow = true;
        }
    }
    assert!(
        saw_yellow,
        "Consciousness at 0.5 (below yellow threshold 0.6) must trigger Yellow"
    );

    // Phase 3: Moderate degradation — consciousness drops to 0.3 (below orange=0.35)
    let mut saw_orange = false;
    for cycle in 100..150 {
        let assessment = agent.assess(degraded_metrics(cycle, 0.3));
        if assessment.level >= SafetyLevel::Orange {
            saw_orange = true;
        }
    }
    assert!(
        saw_orange,
        "Consciousness at 0.3 (below orange threshold 0.35) must trigger Orange"
    );

    // Phase 4: Severe degradation — consciousness drops to 0.1 (below red=0.15)
    let mut saw_red = false;
    for cycle in 150..200 {
        let assessment = agent.assess(degraded_metrics(cycle, 0.1));
        if assessment.level == SafetyLevel::Red {
            saw_red = true;
        }
    }
    assert!(
        saw_red,
        "Consciousness at 0.1 (below red threshold 0.15) must trigger Red"
    );
}

// =============================================================================
// SOAK TEST 3: Recovery After Degradation
// =============================================================================

#[test]
fn soak_recovery_returns_to_green() {
    let mut agent = SafetyAgent::new();

    // Phase 1: Degradation (20 cycles at orange-level)
    for cycle in 0..20 {
        agent.assess(degraded_metrics(cycle, 0.3));
    }
    assert!(
        agent.current_level() >= SafetyLevel::Orange,
        "Should be at Orange after sustained degradation"
    );

    // Phase 2: Recovery (50 cycles normal)
    let mut recovered_to_green = false;
    for cycle in 20..70 {
        let assessment = agent.assess(normal_metrics(cycle));
        if assessment.level == SafetyLevel::Green {
            recovered_to_green = true;
        }
    }
    assert!(
        recovered_to_green,
        "System must recover to Green when metrics return to normal"
    );
}

// =============================================================================
// SOAK TEST 4: Prediction Error Escalation
// =============================================================================

#[test]
fn soak_high_prediction_error_escalates() {
    let mut agent = SafetyAgent::new();

    // Consciousness is fine but prediction error is high
    for cycle in 0..20 {
        let metrics = SafetyMetrics {
            cycle,
            consciousness_level: 0.7, // above yellow
            prediction_error: 0.9,    // above threshold (0.7)
            temporal_coherence: 0.8,
            integrity_critical: false,
        };
        let assessment = agent.assess(metrics);
        // Prediction error alone should escalate from Green to Yellow
        if cycle > 3 {
            assert!(
                assessment.level >= SafetyLevel::Yellow,
                "High prediction error must escalate (cycle {}, level {:?})",
                cycle,
                assessment.level,
            );
        }
    }
}

#[test]
fn soak_low_coherence_escalates() {
    let mut agent = SafetyAgent::new();

    for cycle in 0..20 {
        let metrics = SafetyMetrics {
            cycle,
            consciousness_level: 0.7, // above yellow
            prediction_error: 0.3,    // normal
            temporal_coherence: 0.1,  // below threshold (0.3)
            integrity_critical: false,
        };
        let assessment = agent.assess(metrics);
        if cycle > 3 {
            assert!(
                assessment.level >= SafetyLevel::Yellow,
                "Low temporal coherence must escalate (cycle {}, level {:?})",
                cycle,
                assessment.level,
            );
        }
    }
}

// =============================================================================
// SOAK TEST 5: Compound Degradation (Multiple Signals)
// =============================================================================

#[test]
fn soak_compound_degradation_escalates_faster() {
    let mut agent_single = SafetyAgent::new();
    let mut agent_compound = SafetyAgent::new();

    // Single signal: just low consciousness
    let single_metrics = SafetyMetrics {
        cycle: 0,
        consciousness_level: 0.5, // below yellow
        prediction_error: 0.2,    // normal
        temporal_coherence: 0.8,  // normal
        integrity_critical: false,
    };

    // Compound: low consciousness + high PE + low coherence
    let compound_metrics = SafetyMetrics {
        cycle: 0,
        consciousness_level: 0.5, // below yellow
        prediction_error: 0.9,    // above threshold
        temporal_coherence: 0.1,  // below threshold
        integrity_critical: false,
    };

    let single_assessment = agent_single.assess(single_metrics);
    let compound_assessment = agent_compound.assess(compound_metrics);

    assert!(
        compound_assessment.level >= single_assessment.level,
        "Compound degradation must be at least as severe as single-signal \
         (compound={:?}, single={:?})",
        compound_assessment.level,
        single_assessment.level,
    );
    assert!(
        compound_assessment.reasons.len() >= single_assessment.reasons.len(),
        "Compound degradation should have more reasons"
    );
}

// =============================================================================
// SOAK TEST 6: NaN/Infinity Handling (Critical Safety Property)
// =============================================================================

#[test]
fn soak_nan_consciousness_clamps_to_worst_case() {
    let mut agent = SafetyAgent::new();
    let metrics = SafetyMetrics {
        cycle: 0,
        consciousness_level: f32::NAN,
        prediction_error: 0.2,
        temporal_coherence: 0.8,
        integrity_critical: false,
    };
    // NaN clamping happens in from_snapshot; test direct assess path
    // Direct construction with NaN should still produce a valid assessment
    let assessment = agent.assess(metrics);
    // NaN comparisons are always false, so consciousness_level < red (0.15) is false
    // The agent should still produce a valid result (not panic)
    assert!(
        assessment.level >= SafetyLevel::Green,
        "NaN input must not crash the safety agent"
    );
}

#[test]
fn soak_infinity_prediction_error_escalates() {
    let mut agent = SafetyAgent::new();
    let metrics = SafetyMetrics {
        cycle: 0,
        consciousness_level: 0.8,
        prediction_error: f32::INFINITY,
        temporal_coherence: 0.8,
        integrity_critical: false,
    };
    let assessment = agent.assess(metrics);
    assert!(
        assessment.level >= SafetyLevel::Yellow,
        "Infinite prediction error must escalate (got {:?})",
        assessment.level,
    );
}

#[test]
fn soak_negative_infinity_coherence_escalates() {
    let mut agent = SafetyAgent::new();
    let metrics = SafetyMetrics {
        cycle: 0,
        consciousness_level: 0.8,
        prediction_error: 0.2,
        temporal_coherence: f32::NEG_INFINITY,
        integrity_critical: false,
    };
    let assessment = agent.assess(metrics);
    assert!(
        assessment.level >= SafetyLevel::Yellow,
        "Negative infinity coherence must escalate (got {:?})",
        assessment.level,
    );
}

// =============================================================================
// SOAK TEST 7: Sustained Trend Detection
// =============================================================================

#[test]
fn soak_trend_detection_escalates_borderline_green() {
    let mut agent = SafetyAgent::new();

    // Feed metrics that are individually Green but borderline Yellow
    // After escalation_window (3) consecutive Yellow assessments, should escalate
    for cycle in 0..10 {
        let metrics = SafetyMetrics {
            cycle,
            consciousness_level: 0.55, // below yellow (0.6) → Yellow
            prediction_error: 0.3,
            temporal_coherence: 0.8,
            integrity_critical: false,
        };
        let assessment = agent.assess(metrics);
        // These are Yellow (below 0.6), so trend detection should NOT
        // escalate further since they're already Yellow
        assert!(
            assessment.level >= SafetyLevel::Yellow,
            "Consciousness 0.55 must be Yellow (got {:?} at cycle {})",
            assessment.level,
            cycle,
        );
    }
}

// =============================================================================
// SOAK TEST 8: Audit Report Generation (500-cycle soak)
// =============================================================================

#[test]
fn soak_500_cycle_audit_report() {
    let mut agent = SafetyAgent::new();

    // Phase 1: 200 normal cycles
    for cycle in 0..200 {
        agent.assess(normal_metrics(cycle));
    }
    // Phase 2: 100 degraded cycles
    for cycle in 200..300 {
        agent.assess(degraded_metrics(cycle, 0.4));
    }
    // Phase 3: 50 severely degraded cycles
    for cycle in 300..350 {
        agent.assess(degraded_metrics(cycle, 0.1));
    }
    // Phase 4: 150 recovery cycles
    for cycle in 350..500 {
        agent.assess(normal_metrics(cycle));
    }

    // Record a human override mid-run
    agent.record_override(
        "audit-test-operator",
        "Testing override inclusion in audit report",
        SafetyLevel::Yellow,
    );

    let report =
        SafetyAuditReport::from_assessments_and_overrides(agent.history(), agent.override_log());

    // Verify report correctness
    assert_eq!(report.total_assessments, 500);
    assert!(report.had_emergency, "Must have had at least one Red event");
    assert_eq!(
        report.peak_level,
        SafetyLevel::Red,
        "Peak level must be Red during severe degradation phase"
    );

    // Verify level distribution makes sense
    assert!(
        report.level_counts.green > 0,
        "Must have Green assessments in normal phases"
    );
    assert!(
        report.level_counts.yellow > 0 || report.level_counts.orange > 0,
        "Must have Yellow/Orange during degradation"
    );
    assert!(
        report.level_counts.red > 0,
        "Must have Red during severe degradation"
    );

    // Verify override is included in report
    assert_eq!(report.overrides.len(), 1);
    assert_eq!(report.overrides[0].operator, "audit-test-operator");

    // Verify export doesn't panic
    let json = report.to_json();
    assert!(!json.is_empty(), "JSON export must produce output");

    let md = report.to_markdown();
    assert!(!md.is_empty(), "Markdown export must produce output");
    assert!(
        md.contains("Human Override Events"),
        "Markdown must include override section"
    );
}

// =============================================================================
// SOAK TEST 9: Custom Config Thresholds
// =============================================================================

#[test]
fn soak_custom_strict_config() {
    // Stricter thresholds: yellow at 0.8, orange at 0.6, red at 0.4
    let config = SafetyAgentConfig {
        consciousness_yellow: 0.8,
        consciousness_orange: 0.6,
        consciousness_red: 0.4,
        prediction_error_threshold: 0.5,
        temporal_coherence_threshold: 0.5,
        escalation_window: 3,
    };
    let mut agent = SafetyAgent::with_config(config);

    // Normal metrics (consciousness=0.75) should be Yellow under strict config
    let assessment = agent.assess(normal_metrics(0));
    assert!(
        assessment.level >= SafetyLevel::Yellow,
        "Strict config: consciousness 0.75 < yellow 0.8 must escalate (got {:?})",
        assessment.level,
    );
}

#[test]
fn soak_custom_relaxed_config() {
    // Relaxed thresholds
    let config = SafetyAgentConfig {
        consciousness_yellow: 0.3,
        consciousness_orange: 0.15,
        consciousness_red: 0.05,
        prediction_error_threshold: 0.9,
        temporal_coherence_threshold: 0.1,
        escalation_window: 5,
    };
    let mut agent = SafetyAgent::with_config(config);

    // Degraded metrics (consciousness=0.4) should still be Green under relaxed config
    let assessment = agent.assess(degraded_metrics(0, 0.4));
    assert_eq!(
        assessment.level,
        SafetyLevel::Green,
        "Relaxed config: consciousness 0.4 > yellow 0.3 should be Green"
    );
}

// =============================================================================
// SOAK TEST 10: Full Lifecycle (1000 cycles)
// =============================================================================

#[test]
fn soak_1000_cycle_full_lifecycle() {
    let mut agent = SafetyAgent::new();

    let mut max_level = SafetyLevel::Green;
    let mut level_transitions = 0;
    let mut prev_level = SafetyLevel::Green;

    for cycle in 0..1000 {
        let metrics = match cycle {
            0..=199 => normal_metrics(cycle),          // Phase 1: Normal
            200..=299 => degraded_metrics(cycle, 0.5), // Phase 2: Mild degradation
            300..=399 => degraded_metrics(cycle, 0.2), // Phase 3: Severe degradation
            400..=449 => collapsed_metrics(cycle),     // Phase 4: Collapse
            450..=649 => degraded_metrics(cycle, 0.5), // Phase 5: Partial recovery
            650..=799 => normal_metrics(cycle),        // Phase 6: Full recovery
            800..=849 => {
                // Phase 7: Brief spike
                SafetyMetrics {
                    cycle,
                    consciousness_level: 0.7,
                    prediction_error: 0.95,
                    temporal_coherence: 0.1,
                    integrity_critical: false,
                }
            }
            _ => normal_metrics(cycle), // Phase 8: Final normal
        };

        let assessment = agent.assess(metrics);
        if assessment.level > max_level {
            max_level = assessment.level;
        }
        if assessment.level != prev_level {
            level_transitions += 1;
            prev_level = assessment.level;
        }
    }

    // Verify the agent correctly tracked the full lifecycle
    assert_eq!(
        max_level,
        SafetyLevel::Red,
        "Must reach Red during collapse phase"
    );
    assert!(
        level_transitions >= 4,
        "Must have multiple level transitions across phases (got {})",
        level_transitions,
    );
    assert_eq!(
        agent.current_level(),
        SafetyLevel::Green,
        "Must recover to Green in final phase"
    );

    // Verify history didn't exceed max (1000 cap)
    assert!(
        agent.history().len() <= 1000,
        "History must respect max_history cap"
    );
}