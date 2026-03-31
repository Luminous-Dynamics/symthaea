// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MYCEL reputation page: soulbound score built from peer recognition.
//! AI-interactable: data-mycel-score, data-component breakdown, data-recognition-id.

use leptos::prelude::*;
use crate::contexts::finance_context::use_finance;
use crate::contexts::civic_actions;

#[component]
pub fn RecognitionPage() -> impl IntoView {
    let fin = use_finance();

    let (recognize_did, set_recognize_did) = signal(String::new());
    let (recognize_type, set_recognize_type) = signal("Community".to_string());

    view! {
        <div class="recognition-page" data-page="recognition" role="main">
            <h1 class="page-title">"Reputation"</h1>
            <p class="page-subtitle">"MYCEL — soulbound, earned through care, not purchased"</p>

            // MYCEL Score breakdown
            <section class="mycel-score-section" aria-label="MYCEL reputation score" data-section="mycel-score">
                {move || {
                    let m = fin.mycel_score.get();
                    view! {
                        <div
                            class="mycel-score-card"
                            data-mycel-score=format!("{:.3}", m.score)
                            data-mycel-tier=format!("{:?}", m.tier)
                            data-active-months=m.active_months.to_string()
                        >
                            <div class="mycel-main">
                                <span class="mycel-value" data-metric="score">
                                    {format!("{:.2}", m.score)}
                                </span>
                                <span class=format!("mycel-tier-badge {}", m.tier.css_class())>
                                    {m.tier.label()}
                                </span>
                            </div>
                            <div class="mycel-breakdown" data-component="score-breakdown">
                                <div class="mycel-dim" data-dimension="participation" data-value=format!("{:.2}", m.participation)>
                                    <span class="dim-label">"participation (40%)"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill dim-fill-participation"
                                            style=format!("width: {}%", m.participation * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", m.participation)}</span>
                                </div>
                                <div class="mycel-dim" data-dimension="recognition" data-value=format!("{:.2}", m.recognition)>
                                    <span class="dim-label">"recognition (20%)"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill dim-fill-recognition"
                                            style=format!("width: {}%", m.recognition * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", m.recognition)}</span>
                                </div>
                                <div class="mycel-dim" data-dimension="validation" data-value=format!("{:.2}", m.validation)>
                                    <span class="dim-label">"validation (20%)"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill dim-fill-validation"
                                            style=format!("width: {}%", m.validation * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", m.validation)}</span>
                                </div>
                                <div class="mycel-dim" data-dimension="longevity" data-value=format!("{:.2}", m.longevity)>
                                    <span class="dim-label">"longevity (20%)"</span>
                                    <div class="dim-bar">
                                        <div class="dim-fill dim-fill-longevity"
                                            style=format!("width: {}%", m.longevity * 100.0)></div>
                                    </div>
                                    <span class="dim-value">{format!("{:.2}", m.longevity)}</span>
                                </div>
                            </div>
                            <div class="mycel-tenure">
                                {format!("active for {} months", m.active_months)}
                            </div>
                        </div>
                    }
                }}
            </section>

            // Recognize a peer
            <section class="recognize-section" aria-label="recognize a community member" data-section="recognize-form">
                <h2 class="section-title">"Recognize someone"</h2>
                <form
                    class="recognize-form"
                    data-form="recognize-member"
                    on:submit=move |ev| {
                        ev.prevent_default();
                        if !recognize_did.get_untracked().trim().is_empty() {
                            civic_actions::recognize_member(
                                recognize_did.get_untracked(),
                                recognize_type.get_untracked(),
                            );
                            set_recognize_did.set(String::new());
                        }
                    }
                >
                    <div class="form-field">
                        <label for="recognize-did">"Who deserves recognition?"</label>
                        <input
                            id="recognize-did"
                            type="text"
                            class="form-input"
                            placeholder="did:mycelix:..."
                            data-field="recipient-did"
                            prop:value=move || recognize_did.get()
                            on:input=move |ev| set_recognize_did.set(event_target_value(&ev))
                        />
                    </div>
                    <div class="form-field">
                        <label for="recognize-type">"For what kind of contribution?"</label>
                        <select
                            id="recognize-type"
                            class="form-select"
                            data-field="contribution-type"
                            on:change=move |ev| set_recognize_type.set(event_target_value(&ev))
                        >
                            <option value="Community">"Community"</option>
                            <option value="Care">"Care"</option>
                            <option value="Technical">"Technical"</option>
                            <option value="Governance">"Governance"</option>
                            <option value="Creative">"Creative"</option>
                            <option value="Education">"Education"</option>
                            <option value="General">"General"</option>
                        </select>
                    </div>
                    <button
                        type="submit"
                        class="submit-btn"
                        data-action="recognize-member"
                        disabled=move || recognize_did.get().trim().is_empty()
                    >
                        "recognize this person"
                    </button>
                </form>
            </section>

            // Recognition history
            <section class="recognition-history-section" aria-label="recognitions received" data-section="recognition-history">
                <h2 class="section-title">"Recognitions received"</h2>
                <div class="recognition-list" role="list">
                    {move || {
                        let recs = fin.recognitions_received.get();
                        if recs.is_empty() {
                            view! {
                                <p class="empty-state">"no recognitions yet — community sees what you give"</p>
                            }.into_any()
                        } else {
                            recs.into_iter().map(|r| {
                                let from = r.recognizer_did.split(':').last()
                                    .unwrap_or(&r.recognizer_did).to_string();
                                view! {
                                    <div
                                        class="recognition-card"
                                        data-recognition-id=r.hash.clone()
                                        data-from=r.recognizer_did.clone()
                                        data-type=format!("{:?}", r.contribution_type)
                                        data-weight=format!("{:.2}", r.weight)
                                        data-cycle=r.cycle_id.clone()
                                        role="listitem"
                                    >
                                        <span class="recognition-from">{format!("from {from}")}</span>
                                        <span class="recognition-type">{r.contribution_type.label()}</span>
                                        <span class="recognition-weight">{format!("weight {:.2}", r.weight)}</span>
                                    </div>
                                }
                            }).collect_view().into_any()
                        }
                    }}
                </div>
            </section>
        </div>
    }
}
