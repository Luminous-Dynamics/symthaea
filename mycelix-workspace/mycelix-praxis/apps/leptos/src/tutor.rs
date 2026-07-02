// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resonant Whisper — AI Scaffolding with Sub-Turing Moral Constraints.

use leptos::prelude::*;

#[component]
pub fn ResonantWhisper() -> impl IntoView {
    let (input, set_input) = signal("".to_string());
    let (response, set_response) = signal("Awaiting your inquiry...".to_string());
    
    let ask_tutor = move |_| {
        let text = input.get().to_lowercase();
        
        // MORAL SCAFFOLDING: Sub-Turing Refusal Logic
        let scaffolding_response = if text.contains("write") || text.contains("do") || text.contains("complete") {
            "I cannot complete the work for you, as that would hinder your mastery. However, I can nudge you: What is the primary thermodynamic goal of this project?"
        } else if text.contains("what") || text.contains("how") || text.contains("explain") {
            "Let's look at the first principles. If you were to apply the Principle of Vibration to this mesh network, what frequency would you expect?"
        } else {
            "I am listening. Share your reasoning, and I will scaffold your path."
        };
        
        set_response.set(scaffolding_response.to_string());
    };

    view! {
        <div class="resonant-whisper-box" style="position: fixed; bottom: 80px; right: 20px; width: 320px; background: var(--surface-high); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 10px 25px rgba(0,0,0,0.2); z-index: 1000">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--primary)">"Resonant Whisper"</h4>
            <div class="tutor-response" style="font-size: 0.85rem; line-height: 1.5; color: var(--text-secondary); min-height: 100px">
                {move || response.get()}
            </div>
            
            <div class="tutor-input-area" style="margin-top: 1rem">
                <input 
                    type="text" 
                    placeholder="Inquire with the mesh..." 
                    style="width: 100%; font-size: 0.8rem; padding: 0.5rem; border-radius: 6px; border: 1px solid var(--border); background: var(--surface)"
                    prop:value=input
                    on:input=move |ev| set_input.set(event_target_value(&ev))
                />
                <button class="btn-sm btn-primary" style="width: 100%; margin-top: 0.5rem" on:click=ask_tutor>
                    "Transmit Inquiry"
                </button>
            </div>
            <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.8rem; text-align: center; text-transform: uppercase; letter-spacing: 0.5px">
                "Constraint: Sub-Turing Mentorship Active"
            </div>
        </div>
    }
}
