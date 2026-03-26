// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Consciousness-aware learning engine.
//!
//! Uses Symthaea consciousness metrics (Phi, neuromodulators) to adapt
//! the learning experience in real-time. Decisions include:
//! - When to present new material vs review
//! - When to enforce breaks
//! - What difficulty level to target
//! - Whether to recommend peer work vs solo study
//! - Content presentation mode (visual, auditory, etc.)
//!
//! # Neuromodulator mapping
//!
//! The Spore engine exposes dopamine, serotonin, and norepinephrine.
//! The learning engine derives proxy signals for cortisol, acetylcholine,
//! and oxytocin from these:
//!
//! - **Cortisol proxy**: high norepinephrine + low serotonin indicates stress
//! - **Acetylcholine proxy**: high consciousness level indicates focused attention
//! - **Oxytocin proxy**: high serotonin + moderate dopamine indicates social bonding
//!
//! These are documented approximations, not neuroscience claims.

use leptos::prelude::*;

use crate::consciousness::{use_consciousness, ConsciousnessState};

// Re-use grade adaptation functions. These are in the WASM zome crate,
// so we duplicate the pure logic here for the Leptos client side.
// The canonical source is adaptive_zome/coordinator/src/grade_adaptation.rs.
mod grade_params {
    /// Mastery threshold by grade level (permille).
    pub fn mastery_threshold_for_grade(grade_ordinal: u8) -> u16 {
        match grade_ordinal {
            0..=1 => 900,
            2..=4 => 850,
            5..=7 => 800,
            8..=10 => 800,
            11..=13 => 750,
            _ => 800,
        }
    }

    /// Recommended session length in minutes by grade.
    pub fn session_length_for_grade(grade_ordinal: u8) -> u32 {
        match grade_ordinal {
            0..=1 => 15,
            2..=4 => 25,
            5..=7 => 35,
            8..=10 => 45,
            11..=13 => 55,
            _ => 45,
        }
    }

    /// Break frequency (minutes between breaks) by grade.
    pub fn break_interval_for_grade(grade_ordinal: u8) -> u32 {
        match grade_ordinal {
            0..=1 => 10,
            2..=4 => 20,
            5..=7 => 30,
            8..=10 => 40,
            11..=13 => 50,
            _ => 30,
        }
    }

    /// Zone of Proximal Development range (min, max offset in permille).
    #[allow(dead_code)]
    pub fn zpd_range_for_grade(grade_ordinal: u8) -> (u16, u16) {
        match grade_ordinal {
            0..=1 => (50, 100),
            2..=4 => (75, 150),
            5..=7 => (100, 200),
            8..=10 => (100, 250),
            11..=13 => (100, 300),
            _ => (100, 200),
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Learning readiness based on consciousness state.
#[derive(Clone, Debug, PartialEq)]
pub enum LearningReadiness {
    /// Phi >= 0.55, low cortisol -- optimal learning window.
    FlowState,
    /// Phi >= 0.4, moderate engagement -- standard learning.
    Ready,
    /// Phi < 0.4, low dopamine -- needs warm-up or scaffolding.
    NeedsWarmup,
    /// High cortisol (> 0.65) -- must take a break.
    NeedsBreak,
    /// Very low engagement -- suggest switching activity.
    Disengaged,
}

/// Recommended learning action based on consciousness + grade + mastery.
#[derive(Clone, Debug)]
pub struct LearningRecommendation {
    pub readiness: LearningReadiness,
    pub recommended_action: RecommendedAction,
    /// -1.0 to +1.0 (easier to harder).
    pub difficulty_adjustment: f32,
    pub session_time_remaining_mins: u32,
    pub should_show_hint: bool,
    pub content_mode: ContentMode,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RecommendedAction {
    PresentNewMaterial,
    ReviewWithSRS,
    PeerCollaboration,
    TakeBreak { duration_mins: u32 },
    WarmupActivity,
    DeepPractice,
    CreativeProject,
    SwitchSubject,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ContentMode {
    Visual,
    Auditory,
    ReadWrite,
    Kinesthetic,
    Mixed,
}

// ---------------------------------------------------------------------------
// Neuromodulator proxy derivation
// ---------------------------------------------------------------------------

/// Derived neuromodulator proxies from the three Spore signals.
struct NeuromodProxies {
    dopamine: f32,
    cortisol: f32,
    acetylcholine: f32,
    oxytocin: f32,
}

/// Derive proxy neuromodulators from the available Spore signals.
///
/// - Cortisol: stress proxy = NE * (1 - 5HT). High NE + low serotonin = stress.
/// - Acetylcholine: focus proxy = consciousness level (Phi-like).
/// - Oxytocin: social bonding proxy = 5HT * 0.6 + DA * 0.4 (moderate of both).
fn derive_proxies(state: &ConsciousnessState) -> NeuromodProxies {
    let da = state.neuromod_dopamine;
    let se = state.neuromod_serotonin;
    let ne = state.neuromod_norepinephrine;

    let cortisol = (ne * (1.0 - se)).clamp(0.0, 1.0);
    let acetylcholine = state.consciousness_level.clamp(0.0, 1.0);
    let oxytocin = (se * 0.6 + da * 0.4).clamp(0.0, 1.0);

    NeuromodProxies {
        dopamine: da,
        cortisol,
        acetylcholine,
        oxytocin,
    }
}

// ---------------------------------------------------------------------------
// Core decision function
// ---------------------------------------------------------------------------

/// Compute learning recommendation from consciousness state.
///
/// This is a pure function with no leptos dependencies, making it easy to test.
pub fn compute_recommendation(
    consciousness_level: f32,
    dopamine: f32,
    cortisol: f32,
    acetylcholine: f32,
    oxytocin: f32,
    current_mastery_permille: u16,
    grade_ordinal: u8,
    session_elapsed_mins: u32,
    vark_preference: Option<ContentMode>,
) -> LearningRecommendation {
    use grade_params::*;

    let max_session = session_length_for_grade(grade_ordinal);
    let break_interval = break_interval_for_grade(grade_ordinal);
    let mastery_threshold = mastery_threshold_for_grade(grade_ordinal);

    // Determine readiness
    let readiness = if cortisol > 0.65 {
        LearningReadiness::NeedsBreak
    } else if consciousness_level >= 0.55 && cortisol < 0.4 {
        LearningReadiness::FlowState
    } else if consciousness_level >= 0.4 {
        LearningReadiness::Ready
    } else if dopamine < 0.3 {
        LearningReadiness::Disengaged
    } else {
        LearningReadiness::NeedsWarmup
    };

    // Determine action
    let action = match &readiness {
        LearningReadiness::NeedsBreak => RecommendedAction::TakeBreak {
            duration_mins: if grade_ordinal <= 1 { 10 } else { 5 },
        },
        LearningReadiness::Disengaged => {
            if oxytocin > 0.5 {
                RecommendedAction::PeerCollaboration
            } else {
                RecommendedAction::SwitchSubject
            }
        }
        LearningReadiness::NeedsWarmup => RecommendedAction::WarmupActivity,
        LearningReadiness::FlowState => {
            if current_mastery_permille >= mastery_threshold {
                RecommendedAction::CreativeProject
            } else {
                RecommendedAction::DeepPractice
            }
        }
        LearningReadiness::Ready => {
            if current_mastery_permille < mastery_threshold / 2 {
                RecommendedAction::PresentNewMaterial
            } else {
                RecommendedAction::ReviewWithSRS
            }
        }
    };

    // Time management
    let time_remaining = max_session.saturating_sub(session_elapsed_mins);
    let needs_break = session_elapsed_mins >= break_interval;

    // Difficulty adjustment
    let difficulty_adj = match &readiness {
        LearningReadiness::FlowState => 0.3,
        LearningReadiness::Ready => 0.0,
        LearningReadiness::NeedsWarmup => -0.3,
        _ => -0.5,
    };

    // Hints: show when consciousness is low or dopamine (motivation) is low
    let show_hint = consciousness_level < 0.35 || dopamine < 0.25;

    // Content mode: respect VARK preference, otherwise derive from neuromods
    let mode = vark_preference.unwrap_or_else(|| {
        if acetylcholine > 0.6 {
            ContentMode::ReadWrite // High focus -> reading
        } else if oxytocin > 0.5 {
            ContentMode::Kinesthetic // Social -> hands-on
        } else {
            ContentMode::Mixed
        }
    });

    // Reason string (student-facing)
    let reason = match &readiness {
        LearningReadiness::FlowState => {
            "You're in the zone! Let's go deeper.".to_string()
        }
        LearningReadiness::Ready => {
            "Good focus. Let's learn something new.".to_string()
        }
        LearningReadiness::NeedsWarmup => {
            "Let's warm up with something familiar first.".to_string()
        }
        LearningReadiness::NeedsBreak => {
            "Time for a break \u{2014} your brain needs rest to consolidate.".to_string()
        }
        LearningReadiness::Disengaged => {
            "Let's try something different to re-engage.".to_string()
        }
    };

    // Override action if break interval exceeded (unless already a break)
    let final_action =
        if needs_break && !matches!(action, RecommendedAction::TakeBreak { .. }) {
            RecommendedAction::TakeBreak { duration_mins: 5 }
        } else {
            action
        };

    LearningRecommendation {
        readiness,
        recommended_action: final_action,
        difficulty_adjustment: difficulty_adj,
        session_time_remaining_mins: time_remaining,
        should_show_hint: show_hint,
        content_mode: mode,
        reason,
    }
}

// ---------------------------------------------------------------------------
// Leptos context
// ---------------------------------------------------------------------------

/// The learning engine context -- provides reactive recommendations.
#[derive(Clone)]
pub struct LearningEngineCtx {
    pub recommendation: ReadSignal<LearningRecommendation>,
    pub readiness: ReadSignal<LearningReadiness>,
    pub session_elapsed_mins: ReadSignal<u32>,
    pub total_focus_mins: ReadSignal<u32>,
    pub breaks_taken: ReadSignal<u32>,
}

/// Retrieve the learning engine context from the nearest provider.
///
/// # Panics
/// Panics if called outside a `LearningEngineProvider` subtree.
pub fn use_learning_engine() -> LearningEngineCtx {
    expect_context::<LearningEngineCtx>()
}

/// Default recommendation for initialization.
fn default_recommendation() -> LearningRecommendation {
    LearningRecommendation {
        readiness: LearningReadiness::Ready,
        recommended_action: RecommendedAction::PresentNewMaterial,
        difficulty_adjustment: 0.0,
        session_time_remaining_mins: 45,
        should_show_hint: false,
        content_mode: ContentMode::Mixed,
        reason: "Starting session...".to_string(),
    }
}

/// Provider component that bridges consciousness state to learning decisions.
///
/// Reads from `ConsciousnessCtx` (must be an ancestor) and produces
/// `LearningEngineCtx` for child components. Consciousness ticks at 20Hz
/// but the recommendation is throttled to update at 1Hz to avoid UI thrash.
///
/// # Props
/// - `grade_ordinal`: Grade level (0=PreK, 1=K, 2=Grade1, ..., 13=Grade12).
/// - `current_mastery_permille`: Current mastery (0-1000).
/// - `vark_preference`: Optional VARK learning style preference.
#[component]
pub fn LearningEngineProvider(
    #[prop(default = 5)] grade_ordinal: u8,
    #[prop(default = 500)] current_mastery_permille: u16,
    #[prop(optional)] vark_preference: Option<ContentMode>,
    children: Children,
) -> impl IntoView {
    let consciousness = use_consciousness();

    let (recommendation, set_recommendation) = signal(default_recommendation());
    let (readiness, set_readiness) = signal(LearningReadiness::Ready);
    let (session_elapsed, set_session_elapsed) = signal(0u32);
    let (total_focus, set_total_focus) = signal(0u32);
    let (breaks_taken, set_breaks_taken) = signal(0u32);

    let ctx = LearningEngineCtx {
        recommendation,
        readiness,
        session_elapsed_mins: session_elapsed,
        total_focus_mins: total_focus,
        breaks_taken,
    };

    provide_context(ctx);

    // 1Hz recommendation update via 1-second interval.
    // Also advances session_elapsed_mins every 60 ticks (= 1 minute).
    Effect::new(move |_| {
        use gloo_timers::callback::Interval;

        if !consciousness.running.get() {
            return;
        }

        let vark = vark_preference.clone();

        // Tick counter for minute tracking (1 tick = 1 second at 1Hz)
        let tick_count = std::cell::Cell::new(0u32);

        let _interval = Interval::new(1_000, move || {
            let state = consciousness.state.get();
            let proxies = derive_proxies(&state);

            let elapsed = session_elapsed.get_untracked();

            let rec = compute_recommendation(
                state.consciousness_level,
                proxies.dopamine,
                proxies.cortisol,
                proxies.acetylcholine,
                proxies.oxytocin,
                current_mastery_permille,
                grade_ordinal,
                elapsed,
                vark.clone(),
            );

            set_readiness.set(rec.readiness.clone());
            set_recommendation.set(rec);

            // Advance elapsed minutes every 60 ticks
            let ticks = tick_count.get() + 1;
            tick_count.set(ticks);
            if ticks % 60 == 0 {
                let new_elapsed = elapsed + 1;
                set_session_elapsed.set(new_elapsed);

                // Track focus minutes (when not on break)
                let current_readiness = readiness.get_untracked();
                if current_readiness != LearningReadiness::NeedsBreak {
                    set_total_focus.set(total_focus.get_untracked() + 1);
                }
            }

            // Track break transitions
            let current = readiness.get_untracked();
            if matches!(
                recommendation.get_untracked().recommended_action,
                RecommendedAction::TakeBreak { .. }
            ) && current != LearningReadiness::NeedsBreak
            {
                // Transitioned into a break
                set_breaks_taken.set(breaks_taken.get_untracked() + 1);
            }
        });

        // Keep the interval alive for the duration of this effect.
        std::mem::forget(_interval);
    });

    children()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_state_high_mastery_recommends_creative_project() {
        let rec = compute_recommendation(
            0.7,  // consciousness: high
            0.6,  // dopamine
            0.2,  // cortisol: low
            0.7,  // acetylcholine
            0.5,  // oxytocin
            900,  // mastery: 90% (above threshold for all grades)
            5,    // grade 4
            10,   // 10 min elapsed (under break interval of 30)
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::FlowState);
        assert_eq!(rec.recommended_action, RecommendedAction::CreativeProject);
        assert!(rec.difficulty_adjustment > 0.0, "Flow state should push harder");
    }

    #[test]
    fn test_flow_state_low_mastery_recommends_deep_practice() {
        let rec = compute_recommendation(
            0.7,  // consciousness: high
            0.6,  // dopamine
            0.2,  // cortisol: low
            0.7,  // acetylcholine
            0.5,  // oxytocin
            300,  // mastery: 30% (below threshold)
            5,    // grade 4
            10,   // 10 min elapsed
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::FlowState);
        assert_eq!(rec.recommended_action, RecommendedAction::DeepPractice);
    }

    #[test]
    fn test_high_cortisol_forces_break() {
        let rec = compute_recommendation(
            0.8,  // consciousness: high (doesn't matter)
            0.8,  // dopamine: high
            0.7,  // cortisol: HIGH -- triggers break
            0.8,  // acetylcholine
            0.8,  // oxytocin
            900,  // mastery: high
            5,    // grade 4
            5,    // only 5 min elapsed
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::NeedsBreak);
        assert!(matches!(
            rec.recommended_action,
            RecommendedAction::TakeBreak { .. }
        ));
    }

    #[test]
    fn test_low_dopamine_high_oxytocin_recommends_peer_collaboration() {
        let rec = compute_recommendation(
            0.2,  // consciousness: low
            0.2,  // dopamine: low -- disengaged
            0.3,  // cortisol: moderate
            0.3,  // acetylcholine
            0.6,  // oxytocin: high -- social
            500,  // mastery
            5,    // grade 4
            10,   // 10 min elapsed
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::Disengaged);
        assert_eq!(rec.recommended_action, RecommendedAction::PeerCollaboration);
    }

    #[test]
    fn test_low_dopamine_low_oxytocin_recommends_switch_subject() {
        let rec = compute_recommendation(
            0.2,  // consciousness: low
            0.2,  // dopamine: low -- disengaged
            0.3,  // cortisol: moderate
            0.3,  // acetylcholine
            0.3,  // oxytocin: low -- not social
            500,  // mastery
            5,    // grade 4
            10,   // 10 min elapsed
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::Disengaged);
        assert_eq!(rec.recommended_action, RecommendedAction::SwitchSubject);
    }

    #[test]
    fn test_session_exceeds_break_interval_forces_break() {
        // Grade 4 break interval = 30 min. 35 min elapsed -> forced break.
        let rec = compute_recommendation(
            0.5,  // consciousness: moderate (Ready)
            0.5,  // dopamine
            0.2,  // cortisol: low
            0.5,  // acetylcholine
            0.5,  // oxytocin
            500,  // mastery
            5,    // grade 4 (break_interval = 30)
            35,   // 35 min > 30 min break interval
            None,
        );
        // Readiness is Ready (not NeedsBreak from cortisol), but action overridden
        assert_eq!(rec.readiness, LearningReadiness::Ready);
        assert!(
            matches!(rec.recommended_action, RecommendedAction::TakeBreak { .. }),
            "Should force break when session exceeds break interval"
        );
    }

    #[test]
    fn test_prek_break_duration_is_10_min() {
        let rec = compute_recommendation(
            0.1,  // consciousness: very low
            0.5,  // dopamine
            0.8,  // cortisol: HIGH -- triggers NeedsBreak
            0.3,  // acetylcholine
            0.5,  // oxytocin
            500,  // mastery
            0,    // PreK (grade_ordinal = 0)
            5,    // 5 min elapsed
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::NeedsBreak);
        assert_eq!(
            rec.recommended_action,
            RecommendedAction::TakeBreak { duration_mins: 10 }
        );
    }

    #[test]
    fn test_kindergarten_break_duration_is_10_min() {
        let rec = compute_recommendation(
            0.1,
            0.5,
            0.8,  // cortisol: HIGH
            0.3,
            0.5,
            500,
            1,    // Kindergarten (grade_ordinal = 1)
            5,
            None,
        );
        assert_eq!(
            rec.recommended_action,
            RecommendedAction::TakeBreak { duration_mins: 10 }
        );
    }

    #[test]
    fn test_grade5_break_duration_is_5_min() {
        let rec = compute_recommendation(
            0.1,
            0.5,
            0.8,  // cortisol: HIGH
            0.3,
            0.5,
            500,
            6,    // Grade 5 (grade_ordinal = 6)
            5,
            None,
        );
        assert_eq!(
            rec.recommended_action,
            RecommendedAction::TakeBreak { duration_mins: 5 }
        );
    }

    #[test]
    fn test_ready_low_mastery_presents_new_material() {
        let rec = compute_recommendation(
            0.45, // consciousness: Ready range
            0.5,  // dopamine
            0.2,  // cortisol: low
            0.5,  // acetylcholine
            0.5,  // oxytocin
            200,  // mastery: 20% (well below half of 800 threshold)
            5,    // grade 4
            10,
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::Ready);
        assert_eq!(rec.recommended_action, RecommendedAction::PresentNewMaterial);
    }

    #[test]
    fn test_ready_high_mastery_recommends_review() {
        let rec = compute_recommendation(
            0.45, // consciousness: Ready range
            0.5,  // dopamine
            0.2,  // cortisol: low
            0.5,  // acetylcholine
            0.5,  // oxytocin
            600,  // mastery: 60% (above half of 800 threshold = 400)
            5,    // grade 4
            10,
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::Ready);
        assert_eq!(rec.recommended_action, RecommendedAction::ReviewWithSRS);
    }

    #[test]
    fn test_needs_warmup_state() {
        let rec = compute_recommendation(
            0.3,  // consciousness: below 0.4
            0.5,  // dopamine: above 0.3 (not disengaged)
            0.2,  // cortisol: low (not break)
            0.3,  // acetylcholine
            0.3,  // oxytocin
            500,
            5,
            10,
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::NeedsWarmup);
        assert_eq!(rec.recommended_action, RecommendedAction::WarmupActivity);
        assert!(rec.difficulty_adjustment < 0.0, "Warmup should reduce difficulty");
    }

    #[test]
    fn test_hint_shown_at_low_consciousness() {
        let rec = compute_recommendation(
            0.3,  // consciousness: low (< 0.35)
            0.5,  // dopamine: above hint threshold
            0.2,
            0.3,
            0.3,
            500,
            5,
            10,
            None,
        );
        assert!(rec.should_show_hint, "Should show hint when consciousness < 0.35");
    }

    #[test]
    fn test_hint_shown_at_low_dopamine() {
        let rec = compute_recommendation(
            0.5,  // consciousness: above hint threshold
            0.2,  // dopamine: low (< 0.25)
            0.2,
            0.5,
            0.3,
            500,
            5,
            10,
            None,
        );
        assert!(rec.should_show_hint, "Should show hint when dopamine < 0.25");
    }

    #[test]
    fn test_no_hint_when_engaged() {
        let rec = compute_recommendation(
            0.6,  // consciousness: high
            0.6,  // dopamine: high
            0.2,
            0.6,
            0.5,
            500,
            5,
            10,
            None,
        );
        assert!(!rec.should_show_hint, "Should not show hint when engaged");
    }

    #[test]
    fn test_vark_preference_respected() {
        let rec = compute_recommendation(
            0.5,
            0.5,
            0.2,
            0.8, // high acetylcholine would normally -> ReadWrite
            0.3,
            500,
            5,
            10,
            Some(ContentMode::Auditory),
        );
        assert_eq!(rec.content_mode, ContentMode::Auditory);
    }

    #[test]
    fn test_high_acetylcholine_defaults_to_readwrite() {
        let rec = compute_recommendation(
            0.5,
            0.5,
            0.2,
            0.7, // acetylcholine > 0.6
            0.3, // oxytocin <= 0.5
            500,
            5,
            10,
            None,
        );
        assert_eq!(rec.content_mode, ContentMode::ReadWrite);
    }

    #[test]
    fn test_high_oxytocin_defaults_to_kinesthetic() {
        let rec = compute_recommendation(
            0.5,
            0.5,
            0.2,
            0.4, // acetylcholine <= 0.6
            0.6, // oxytocin > 0.5
            500,
            5,
            10,
            None,
        );
        assert_eq!(rec.content_mode, ContentMode::Kinesthetic);
    }

    #[test]
    fn test_session_time_remaining_calculation() {
        // Grade 4 session length = 35 min. 20 min elapsed -> 15 remaining.
        let rec = compute_recommendation(
            0.5, 0.5, 0.2, 0.5, 0.5,
            500, 5, 20,
            None,
        );
        assert_eq!(rec.session_time_remaining_mins, 15);
    }

    #[test]
    fn test_session_time_remaining_does_not_underflow() {
        // Grade PreK session = 15 min. 100 min elapsed -> 0 remaining.
        let rec = compute_recommendation(
            0.5, 0.5, 0.2, 0.5, 0.5,
            500, 0, 100,
            None,
        );
        assert_eq!(rec.session_time_remaining_mins, 0);
    }

    #[test]
    fn test_neuromod_proxy_cortisol_high_stress() {
        let state = ConsciousnessState {
            neuromod_norepinephrine: 0.9,
            neuromod_serotonin: 0.1,
            ..Default::default()
        };
        let proxies = derive_proxies(&state);
        // cortisol = NE * (1 - 5HT) = 0.9 * 0.9 = 0.81
        assert!(proxies.cortisol > 0.65, "High NE + low 5HT should indicate stress");
    }

    #[test]
    fn test_neuromod_proxy_cortisol_low_stress() {
        let state = ConsciousnessState {
            neuromod_norepinephrine: 0.2,
            neuromod_serotonin: 0.8,
            ..Default::default()
        };
        let proxies = derive_proxies(&state);
        // cortisol = 0.2 * (1 - 0.8) = 0.04
        assert!(proxies.cortisol < 0.1, "Low NE + high 5HT should indicate calm");
    }

    #[test]
    fn test_neuromod_proxy_oxytocin() {
        let state = ConsciousnessState {
            neuromod_serotonin: 0.8,
            neuromod_dopamine: 0.6,
            ..Default::default()
        };
        let proxies = derive_proxies(&state);
        // oxytocin = 5HT * 0.6 + DA * 0.4 = 0.48 + 0.24 = 0.72
        assert!(
            (proxies.oxytocin - 0.72).abs() < 0.01,
            "Oxytocin proxy should be weighted blend of 5HT and DA"
        );
    }

    #[test]
    fn test_neuromod_proxy_acetylcholine_tracks_consciousness() {
        let state = ConsciousnessState {
            consciousness_level: 0.75,
            ..Default::default()
        };
        let proxies = derive_proxies(&state);
        assert!(
            (proxies.acetylcholine - 0.75).abs() < 0.01,
            "Acetylcholine proxy should track consciousness level"
        );
    }

    #[test]
    fn test_difficulty_adjustment_ranges() {
        // FlowState -> +0.3
        let flow = compute_recommendation(0.7, 0.6, 0.2, 0.7, 0.5, 300, 5, 10, None);
        assert!((flow.difficulty_adjustment - 0.3).abs() < f32::EPSILON);

        // Ready -> 0.0
        let ready = compute_recommendation(0.45, 0.5, 0.2, 0.5, 0.5, 500, 5, 10, None);
        assert!((ready.difficulty_adjustment - 0.0).abs() < f32::EPSILON);

        // NeedsWarmup -> -0.3
        let warmup = compute_recommendation(0.3, 0.5, 0.2, 0.3, 0.3, 500, 5, 10, None);
        assert!((warmup.difficulty_adjustment - (-0.3)).abs() < f32::EPSILON);

        // NeedsBreak -> -0.5
        let brk = compute_recommendation(0.5, 0.5, 0.8, 0.5, 0.5, 500, 5, 5, None);
        assert!((brk.difficulty_adjustment - (-0.5)).abs() < f32::EPSILON);

        // Disengaged -> -0.5
        let dis = compute_recommendation(0.2, 0.2, 0.3, 0.3, 0.3, 500, 5, 10, None);
        assert!((dis.difficulty_adjustment - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cortisol_takes_priority_over_flow() {
        // Even with high consciousness, high cortisol forces break
        let rec = compute_recommendation(
            0.9,  // consciousness: very high
            0.9,  // dopamine: very high
            0.7,  // cortisol: HIGH -- overrides everything
            0.9,
            0.9,
            900,
            5,
            5,
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::NeedsBreak);
    }

    #[test]
    fn test_grade12_mastery_threshold() {
        // Grade 12 threshold = 750. Mastery 800 = above threshold -> CreativeProject in flow
        let rec = compute_recommendation(
            0.7, 0.6, 0.2, 0.7, 0.5,
            800, 13, 10,
            None,
        );
        assert_eq!(rec.readiness, LearningReadiness::FlowState);
        assert_eq!(rec.recommended_action, RecommendedAction::CreativeProject);
    }
}
