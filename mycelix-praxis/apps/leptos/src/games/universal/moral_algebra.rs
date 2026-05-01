// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Moral Algebra Interactive Lesson — Operationalizing the 16 Obligations.
//! Teaches students how to score and align standards against core axioms.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ObligationState {
    pub name: &'static str,
    pub description: &'static str,
    pub value: u16, // 0-1000
    pub is_perfect: bool,
}

#[component]
pub fn MoralAlgebraLesson() -> impl IntoView {
    let (obligations, set_obligations) = signal(vec![
        ObligationState { name: "Ahimsa", description: "Non-harm and reversibility of actions.", value: 850, is_perfect: true },
        ObligationState { name: "Reciprocity", description: "Sacred reciprocity and mutual exchange.", value: 700, is_perfect: true },
        ObligationState { name: "Truthfulness", description: "Epistemic humility and honest signaling.", value: 900, is_perfect: true },
        ObligationState { name: "Sovereignty", description: "Respect for individual and collective agency.", value: 800, is_perfect: true },
        ObligationState { name: "Eco-Stewardship", description: "Active care for the living substrate.", value: 600, is_perfect: false },
        ObligationState { name: "Ubuntu", description: "I am because we are (Mentorship).", value: 750, is_perfect: false },
    ]);

    let aggregate_alignment = Memo::new(move |_| {
        let obs = obligations.get();
        let sum: u32 = obs.iter().map(|o| o.value as u32).sum();
        (sum / obs.len() as u32) as u16
    });

    view! {
        <div class="moral-algebra-lesson">
            <header class="lesson-header">
                <h3>"SYM-502: Moral Algebra & Alignment"</h3>
                <div class="alignment-score">
                    <span class="score-label">"Aggregate Alignment:"</span>
                    <span class="score-value">{move || aggregate_alignment.get() / 10}"%"</span>
                </div>
            </header>

            <div class="lesson-layout">
                <div class="obligations-editor">
                    <h4>"The 16 Obligations (Ref. Test)"</h4>
                    <p class="instruction">"Adjust the alignment values to see how they impact the standard's promotion threshold."</p>
                    
                    <div class="obligations-list">
                        {move || obligations.get().into_iter().enumerate().map(|(idx, obj)| {
                            let name = obj.name;
                            let desc = obj.description;
                            view! {
                                <div class="obligation-row">
                                    <div class="obj-meta">
                                        <span class=move || if obj.is_perfect { "badge-perfect" } else { "badge-imperfect" }>
                                            {if obj.is_perfect { "PERFECT" } else { "IMPERFECT" }}
                                        </span>
                                        <strong>{name}</strong>
                                    </div>
                                    <p class="obj-desc">{desc}</p>
                                    <input 
                                        type="range" min="0" max="1000" step="10"
                                        prop:value=obj.value
                                        on:input=move |ev| {
                                            let val = event_target_value(&ev).parse().unwrap_or(0);
                                            set_obligations.update(|obs| obs[idx].value = val);
                                        }
                                    />
                                </div>
                            }
                        }).collect_view()}
                    </div>
                </div>

                <div class="alignment-visualization">
                    <h4>"Geometric Radar"</h4>
                    <div class="radar-placeholder">
                        // In a full implementation, this would be an SVG radar chart
                        <div class="radar-circle">
                            <div class="radar-fill" style=move || format!("transform: scale({})", aggregate_alignment.get() as f32 / 1000.0)></div>
                        </div>
                    </div>
                    
                    <div class="audit-prediction">
                        <h5>"Symthaea Auditor Forecast:"</h5>
                        {move || {
                            let score = aggregate_alignment.get();
                            if score > 800 {
                                view! { <div class="status-pass">"\u{2705} Likely to be Promoted"</div> }.into_any()
                            } else if score > 500 {
                                view! { <div class="status-warn">"\u{26A0} Requires Peer Review"</div> }.into_any()
                            } else {
                                view! { <div class="status-fail">"\u{274C} Mathematically Rejected"</div> }.into_any()
                            }
                        }}
                    </div>
                </div>
            </div>

            <footer class="lesson-footer">
                <button class="btn-action success">"Submit Alignment Artifact"</button>
            </footer>
        </div>
    }
}
