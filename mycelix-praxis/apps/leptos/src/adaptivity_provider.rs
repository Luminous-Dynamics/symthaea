// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Adaptivity provider: bridges consciousness -> cognitive adaptivity -> UI.
//!
//! Reads live consciousness signals, derives a `CognitiveState`, runs the
//! sovereignty-aware `adapt_content()` engine, and exposes reactive signals
//! that review pages, dashboards, and overlays can read.

use leptos::prelude::*;

use crate::cognitive_adaptivity::*;
use crate::consciousness::use_consciousness;
use crate::persistence;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Reactive adaptivity context, provided inside `AdaptivityProvider`.
#[derive(Clone)]
pub struct AdaptivityCtx {
    /// Current content adaptation decision.
    pub adaptation: Signal<ContentAdaptation>,
    /// Current sovereignty level.
    pub sovereignty: Signal<SovereigntyLevel>,
    /// Active suggestion (if any) -- `None` when dismissed or nothing to show.
    pub active_suggestion: Signal<Option<Suggestion>>,
    /// Current cognitive state derived from consciousness.
    pub cognitive_state: Signal<CognitiveState>,
    /// Number of consecutive incorrect answers.
    pub consecutive_failures: Signal<u32>,
    /// Number of consecutive correct answers.
    pub consecutive_correct: Signal<u32>,
    /// Recent accuracy (rolling window of last 10 attempts).
    pub recent_accuracy: Signal<f32>,
    /// Pending metacognitive prompt (if any).
    pub metacognitive_prompt_text: Signal<Option<String>>,

    // -- write handles for UI actions --
    set_current_skill: WriteSignal<String>,
    set_mastery: WriteSignal<u16>,
    set_record_attempt: WriteSignal<Option<bool>>,
    set_dismiss_suggestion: WriteSignal<u64>,
    set_accept_suggestion: WriteSignal<Option<ChoiceAction>>,
    set_dismiss_metacognitive: WriteSignal<u64>,
    set_sovereignty: WriteSignal<SovereigntyLevel>,
}

impl AdaptivityCtx {
    /// Record a learning attempt (correct or incorrect).
    /// Drives adaptivity: updates consecutive counts, accuracy, and
    /// may generate a metacognitive prompt or sovereignty event.
    pub fn record_attempt(&self, correct: bool) {
        self.set_record_attempt.set(Some(correct));
    }

    /// Set the current skill being practiced and its mastery level.
    pub fn set_skill(&self, skill: &str, mastery_permille: u16) {
        self.set_current_skill.set(skill.to_string());
        self.set_mastery.set(mastery_permille);
    }

    /// Student dismissed a suggestion ("no thanks").
    pub fn dismiss_suggestion(&self) {
        self.set_dismiss_suggestion.update(|v| *v = v.wrapping_add(1));
    }

    /// Student accepted a suggestion with a specific action.
    pub fn accept_suggestion(&self, action: ChoiceAction) {
        self.set_accept_suggestion.set(Some(action));
    }

    /// Dismiss the current metacognitive prompt.
    pub fn dismiss_metacognitive(&self) {
        self.set_dismiss_metacognitive.update(|v| *v = v.wrapping_add(1));
    }

    /// Enter sandbox mode — no tracking, no adaptation, just exploration.
    pub fn enter_sandbox(&self) {
        self.set_sovereignty.update(|s| s.enter_sandbox());
        crate::persistence::save("praxis_sovereignty", &self.sovereignty.get());
    }

    /// Leave sandbox mode — return to normal.
    pub fn leave_sandbox(&self) {
        self.set_sovereignty.update(|s| s.leave_sandbox());
        crate::persistence::save("praxis_sovereignty", &self.sovereignty.get());
    }

    /// Student requests more support (lowers effective mode by one level).
    pub fn request_support(&self) {
        self.set_sovereignty.update(|s| s.request_support());
        crate::persistence::save("praxis_sovereignty", &self.sovereignty.get());
    }

    /// Student dismisses extra support.
    pub fn dismiss_support(&self) {
        self.set_sovereignty.update(|s| s.dismiss_support());
        crate::persistence::save("praxis_sovereignty", &self.sovereignty.get());
    }

    /// Is sandbox mode active?
    pub fn is_sandbox(&self) -> bool {
        self.sovereignty.get().sandbox_active
    }
}

/// Retrieve the adaptivity context from the nearest provider.
///
/// # Panics
/// Panics if called outside an `AdaptivityProvider` subtree.
pub fn use_adaptivity() -> AdaptivityCtx {
    expect_context::<AdaptivityCtx>()
}

// ---------------------------------------------------------------------------
// Default adaptation (for initialization)
// ---------------------------------------------------------------------------

fn default_adaptation() -> ContentAdaptation {
    ContentAdaptation {
        text_complexity: TextComplexity::Standard,
        modality: Modality::MultiModal,
        difficulty_delta: 0.0,
        suggestion: None,
        peer_suggestion: None,
        reasoning: "Initializing...".into(),
    }
}

fn default_cognitive_state() -> CognitiveState {
    CognitiveState {
        phi: 0.0,
        free_energy: 1.0,
        valence: 0.0,
        arousal: 0.3,
        focus: 0.0,
        motivation: 0.5,
        stress: 0.0,
        social_readiness: 0.5,
        permeability: 0.5,
        motor_command: MotorCommand::NoOp,
    }
}

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

/// Wraps children with an `AdaptivityCtx` that bridges consciousness signals
/// to the sovereignty-aware adaptivity engine.
///
/// Must be nested inside `ConsciousnessProvider`.
#[component]
pub fn AdaptivityProvider(children: Children) -> impl IntoView {
    let consciousness = use_consciousness();

    // -- Core state signals --
    let initial_sovereignty = persistence::load::<SovereigntyLevel>("praxis_sovereignty")
        .unwrap_or_else(SovereigntyLevel::new);
    let (sovereignty, set_sovereignty) = signal(initial_sovereignty);
    let (current_skill, set_current_skill) = signal("multiplication".to_string());
    let (mastery, set_mastery) = signal(500u16);
    let (consecutive_failures, set_consecutive_failures) = signal(0u32);
    let (consecutive_correct, set_consecutive_correct) = signal(0u32);
    let (recent_attempts, set_recent_attempts) = signal(vec![true; 5]);
    let (active_suggestion, set_active_suggestion) = signal::<Option<Suggestion>>(None);
    let (metacognitive_text, set_metacognitive_text) = signal::<Option<String>>(None);

    // -- UI action triggers --
    let (record_attempt_trigger, set_record_attempt) = signal::<Option<bool>>(None);
    let (dismiss_suggestion_counter, set_dismiss_suggestion) = signal(0u64);
    let (accept_suggestion_trigger, set_accept_suggestion) = signal::<Option<ChoiceAction>>(None);
    let (dismiss_metacognitive_counter, set_dismiss_metacognitive) = signal(0u64);

    // Track when we last dismissed a suggestion so we don't immediately re-show
    let (last_dismiss_cycle, set_last_dismiss_cycle) = signal(0u64);

    // -- Derived: recent accuracy --
    let recent_accuracy = Memo::new(move |_| {
        let attempts = recent_attempts.get();
        if attempts.is_empty() {
            return 1.0_f32;
        }
        let correct = attempts.iter().filter(|&&a| a).count() as f32;
        correct / attempts.len() as f32
    });

    // -- Derived: cognitive state (from consciousness) --
    let cognitive_state = Memo::new(move |_| {
        let s = consciousness.state.get();
        derive_cognitive_state(
            s.phi,
            s.free_energy,
            s.neuromod_dopamine,
            s.neuromod_serotonin,
            s.neuromod_norepinephrine,
            MotorCommand::NoOp,
        )
    });

    // -- Derived: content adaptation --
    let adaptation = Memo::new(move |_| {
        let cog = cognitive_state.get();
        let sov = sovereignty.get();
        let skill = current_skill.get();
        let mast = mastery.get();
        let acc = recent_accuracy.get();
        let fails = consecutive_failures.get();

        adapt_content(
            &cog,
            &sov,
            &skill,
            mast,
            acc,
            fails,
            5, // grade_ordinal default (grade 4)
            &[], // student_interests (empty for now)
            DEFAULT_SAFETY_THRESHOLD,
        )
    });

    // -- Effect: handle record_attempt actions --
    Effect::new(move |_| {
        let Some(correct) = record_attempt_trigger.get() else {
            return;
        };
        // Reset the trigger
        set_record_attempt.set(None);

        // Update consecutive counts
        if correct {
            set_consecutive_failures.set(0);
            set_consecutive_correct.update(|n| *n += 1);
        } else {
            set_consecutive_correct.set(0);
            set_consecutive_failures.update(|n| *n += 1);
        }

        // Update rolling accuracy window (keep last 10)
        set_recent_attempts.update(|attempts| {
            attempts.push(correct);
            if attempts.len() > 10 {
                attempts.remove(0);
            }
        });

        // Generate metacognitive prompt
        let sov = sovereignty.get_untracked();
        let skill = current_skill.get_untracked();
        let consec_correct = if correct {
            consecutive_correct.get_untracked()
        } else {
            0
        };
        let prompt = metacognitive_prompt(&sov, correct, consec_correct, &skill);
        set_metacognitive_text.set(prompt);

        // Sovereignty growth: perseverance after failures
        if correct && consecutive_failures.get_untracked() >= 2 {
            set_sovereignty.update(|s| {
                s.record_growth(
                    SovereigntyGrowthType::PerseveranceSuccess,
                    "Pushed through difficulty and got it right".into(),
                    js_sys::Date::now() as u64,
                );
            });
        }
    });

    // -- Effect: persist sovereignty to localStorage on every change --
    Effect::new(move |_| {
        let sov = sovereignty.get();
        persistence::save("praxis_sovereignty", &sov);
    });

    // -- Effect: handle suggestion dismiss --
    Effect::new(move |_| {
        let _ = dismiss_suggestion_counter.get();
        set_active_suggestion.set(None);
        let cycle = consciousness.state.get_untracked().cycle_count;
        set_last_dismiss_cycle.set(cycle);
    });

    // -- Effect: handle suggestion accept --
    Effect::new(move |_| {
        let Some(action) = accept_suggestion_trigger.get() else {
            return;
        };
        set_accept_suggestion.set(None);
        set_active_suggestion.set(None);

        // Sovereignty growth for independent success when declining
        if action == ChoiceAction::KeepGoing {
            // If they chose to keep going, check next attempt for IndependentSuccess
            // (this is tracked by the consecutive_correct -> sovereignty effect above)
        }

        // If they chose a break, record self-regulation
        if matches!(action, ChoiceAction::TakeBreak { .. }) {
            set_sovereignty.update(|s| {
                s.record_growth(
                    SovereigntyGrowthType::SelfRegulatedBreak,
                    "Chose to take a break when it was offered".into(),
                    js_sys::Date::now() as u64,
                );
            });
        }
    });

    // -- Effect: handle metacognitive dismiss --
    Effect::new(move |_| {
        let _ = dismiss_metacognitive_counter.get();
        set_metacognitive_text.set(None);
    });

    // -- Effect: surface suggestions from adaptation (throttled) --
    Effect::new(move |_| {
        let adapt = adaptation.get();
        let cycle = consciousness.state.get().cycle_count;
        let dismiss_cycle = last_dismiss_cycle.get_untracked();

        // Don't show a new suggestion within 40 cycles (~2s) of dismissal
        if cycle.saturating_sub(dismiss_cycle) < 40 {
            return;
        }

        if let Some(suggestion) = adapt.suggestion {
            set_active_suggestion.set(Some(suggestion));
        }
    });

    // -- Provide context --
    let ctx = AdaptivityCtx {
        adaptation: Signal::from(adaptation),
        sovereignty: sovereignty.into(),
        active_suggestion: active_suggestion.into(),
        cognitive_state: Signal::from(cognitive_state),
        consecutive_failures: consecutive_failures.into(),
        consecutive_correct: consecutive_correct.into(),
        recent_accuracy: Signal::from(recent_accuracy),
        metacognitive_prompt_text: metacognitive_text.into(),

        set_current_skill,
        set_mastery,
        set_record_attempt,
        set_dismiss_suggestion,
        set_accept_suggestion,
        set_dismiss_metacognitive,
        set_sovereignty,
    };

    provide_context(ctx);
    children()
}

/// Safety threshold constant — exposed so adapt_content can use it.
const DEFAULT_SAFETY_THRESHOLD: f32 = 0.85;
