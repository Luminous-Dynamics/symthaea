// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Suggestion overlay component.
//!
//! A non-blocking card that slides in from the bottom when the adaptivity
//! engine has a suggestion. Respects sovereignty mode:
//! - **Guardian**: shows the suggestion with an explanation and choices.
//! - **Guide**: shows multiple choice options.
//! - **Mirror**: shows cognitive state data only.
//! - **Autonomous**: hidden (no unsolicited suggestions).

use leptos::prelude::*;

use crate::adaptivity_provider::use_adaptivity;
use crate::cognitive_adaptivity::*;

/// A gentle, non-blocking suggestion card that slides in from the bottom.
///
/// The student can continue working while the suggestion is visible.
/// Include this component on any page that should show sovereignty-aware
/// suggestions (review, dashboard, etc.).
#[component]
pub fn SuggestionOverlay() -> impl IntoView {
    let ctx = use_adaptivity();

    // Extract signals for closure capture (signals are Copy)
    let active_suggestion = ctx.active_suggestion;
    let sovereignty_sig = ctx.sovereignty;
    let metacognitive_sig = ctx.metacognitive_prompt_text;

    // Clone ctx for the button handlers inside each closure
    let ctx_for_buttons = ctx.clone();
    let ctx_for_meta = ctx;

    view! {
        // -- Active suggestion overlay --
        {move || {
            let suggestion = active_suggestion.get();
            let sovereignty = sovereignty_sig.get();
            let mode = sovereignty.mode();

            match suggestion {
                None => view! { <div class="suggestion-overlay hidden"></div> }.into_any(),
                Some(suggestion) => {
                    let is_safety = suggestion.is_safety_concern;
                    let css_class = format!(
                        "suggestion-overlay visible {} {}",
                        match mode {
                            InteractionMode::Guardian => "mode-guardian",
                            InteractionMode::Guide => "mode-guide",
                            InteractionMode::Mirror => "mode-mirror",
                            InteractionMode::Autonomous => "mode-autonomous",
                        },
                        if is_safety { "safety-concern" } else { "" },
                    );

                    let message = suggestion.message.clone();
                    let options = suggestion.options.clone();
                    let ctx_inner = ctx_for_buttons.clone();

                    view! {
                        <div class=css_class>
                            <div class="suggestion-content">
                                // Mode indicator
                                <div class="suggestion-mode-badge">
                                    {match mode {
                                        InteractionMode::Guardian => "Helper",
                                        InteractionMode::Guide => "Guide",
                                        InteractionMode::Mirror => "Mirror",
                                        InteractionMode::Autonomous => "",
                                    }}
                                </div>

                                // Message
                                <p class="suggestion-message">{message}</p>

                                // Choice buttons
                                <div class="suggestion-choices">
                                    {options.into_iter().map(|choice| {
                                        let label = choice.label.clone();
                                        let action = choice.action.clone();
                                        let is_decline = choice.is_decline;
                                        let btn_class = if is_decline {
                                            "suggestion-btn suggestion-btn-decline"
                                        } else {
                                            "suggestion-btn suggestion-btn-accept"
                                        };
                                        let ctx_btn = ctx_inner.clone();

                                        view! {
                                            <button
                                                class=btn_class
                                                on:click=move |_| {
                                                    if is_decline {
                                                        ctx_btn.dismiss_suggestion();
                                                    } else {
                                                        ctx_btn.accept_suggestion(action.clone());
                                                    }
                                                }
                                            >
                                                {label}
                                            </button>
                                        }
                                    }).collect_view()}
                                </div>

                                // Safety note (only shown when teacher-configured threshold triggered)
                                {if is_safety {
                                    view! {
                                        <p class="suggestion-safety-note">
                                            <small>"Your teacher set this check-in to help you stay comfortable."</small>
                                        </p>
                                    }.into_any()
                                } else {
                                    view! { <span></span> }.into_any()
                                }}
                            </div>
                        </div>
                    }.into_any()
                }
            }
        }}

        // -- Metacognitive prompt overlay --
        {move || {
            match metacognitive_sig.get() {
                None => view! { <div class="metacognitive-overlay hidden"></div> }.into_any(),
                Some(prompt) => {
                    let ctx_meta = ctx_for_meta.clone();
                    view! {
                        <div class="metacognitive-overlay visible">
                            <div class="metacognitive-content">
                                <p class="metacognitive-prompt">{prompt}</p>
                                <button
                                    class="metacognitive-dismiss"
                                    on:click=move |_| ctx_meta.dismiss_metacognitive()
                                >
                                    "Got it!"
                                </button>
                            </div>
                        </div>
                    }.into_any()
                }
            }
        }}
    }
}

/// A compact cognitive state display for Mirror mode.
/// Shows the student their own cognitive metrics in kid-friendly terms.
#[component]
pub fn CognitiveStateMirror() -> impl IntoView {
    let ctx = use_adaptivity();
    let cognitive_state = ctx.cognitive_state;
    let sovereignty_sig = ctx.sovereignty;

    view! {
        <div class="cognitive-mirror">
            {move || {
                let cog = cognitive_state.get();
                let sov = sovereignty_sig.get();

                // Only show in Mirror or Autonomous mode
                if !matches!(sov.mode(), InteractionMode::Mirror | InteractionMode::Autonomous) {
                    return view! { <div class="cognitive-mirror-hidden"></div> }.into_any();
                }

                let focus_label = if cog.focus > 0.7 { "High focus" }
                    else if cog.focus > 0.4 { "Good focus" }
                    else { "Low focus" };
                let energy_label = if cog.stress > 0.6 { "Stressed" }
                    else if cog.motivation > 0.6 { "Motivated" }
                    else if cog.motivation < 0.3 { "Low energy" }
                    else { "Steady" };
                let social_label = if cog.social_readiness > 0.6 { "Social" }
                    else { "Solo" };

                view! {
                    <div class="cognitive-mirror-display">
                        <span class="mirror-label">"Your State:"</span>
                        <span class="mirror-tag focus-tag">{focus_label}</span>
                        <span class="mirror-tag energy-tag">{energy_label}</span>
                        <span class="mirror-tag social-tag">{social_label}</span>
                    </div>
                }.into_any()
            }}
        </div>
    }
}
