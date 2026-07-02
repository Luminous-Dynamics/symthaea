// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::context::use_finance_context;
use leptos::prelude::*;

#[component]
pub fn RecognitionPage() -> impl IntoView {
    let ctx = use_finance_context();

    view! {
        <div class="page-recognition">
            <h1>"Recognition"</h1>
            <p class="subtitle">"Peer recognition builds MYCEL reputation."</p>

            <div class="recognition-list">
                {move || ctx.recognitions.get().iter().map(|r| {
                    let from = r.recognizer_did.clone();
                    let ctype = r.contribution_type.label();
                    let weight = r.weight;
                    let cycle = r.cycle_id.clone();
                    view! {
                        <div class="recognition-card">
                            <div class="recog-header">
                                <span class="recog-type">{ctype}</span>
                                <span class="recog-weight">{format!("weight: {weight:.1}")}</span>
                            </div>
                            <span class="recog-from">{format!("From: {from}")}</span>
                            <span class="recog-cycle">{format!("Cycle: {cycle}")}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
