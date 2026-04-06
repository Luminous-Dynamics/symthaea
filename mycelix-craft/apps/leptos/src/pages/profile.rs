// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profile page — craft identity, work history, credentials, and endorsements.
//! Uses mock-first pattern: renders from localStorage immediately, attempts
//! conductor sync after connection.

use leptos::prelude::*;
use mycelix_leptos_core::{
    toasts::{use_toasts, ToastKind},
    loading::LoadingSkeleton,
};

use crate::persistence;

const PROFILE_KEY: &str = "craft_profile_draft";

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct ProfileDraft {
    display_name: String,
    headline: String,
    bio: String,
    location: String,
    website: String,
}

#[component]
pub fn ProfilePage() -> impl IntoView {
    let initial = persistence::load::<ProfileDraft>(PROFILE_KEY).unwrap_or_default();
    let (draft, set_draft) = signal(initial);
    let (saving, set_saving) = signal(false);

    // Persist to localStorage on every change
    Effect::new(move |_| {
        persistence::save(PROFILE_KEY, &draft.get());
    });

    // Save handler — tries conductor first, falls back to localStorage
    let on_save = move |_| {
        set_saving.set(true);
        let toasts = use_toasts();
        // For now, localStorage is the only persistence. When conductor
        // is connected, this will call craft_graph::set_profile via zome call.
        toasts.push("Profile saved locally. Connect to conductor to publish to network.", ToastKind::Info);
        set_saving.set(false);
    };

    view! {
        <div class="page profile-page">
            <h1>"Craft Profile"</h1>

            <div class="profile-grid">
                <div class="profile-section">
                    <h3>"Identity"</h3>
                    <div class="form-group">
                        <label for="display_name">"Display Name"</label>
                        <input
                            id="display_name"
                            type="text"
                            class="form-input"
                            placeholder="Your name"
                            prop:value=move || draft.get().display_name
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_draft.update(|d| d.display_name = val);
                            }
                        />
                    </div>
                    <div class="form-group">
                        <label for="headline">"Headline"</label>
                        <input
                            id="headline"
                            type="text"
                            class="form-input"
                            placeholder="e.g., Systems Architect | Holochain Developer"
                            prop:value=move || draft.get().headline
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_draft.update(|d| d.headline = val);
                            }
                        />
                    </div>
                    <div class="form-group">
                        <label for="bio">"Bio"</label>
                        <textarea
                            id="bio"
                            rows="4"
                            class="form-textarea"
                            placeholder="Tell your story..."
                            prop:value=move || draft.get().bio
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlTextAreaElement>().unwrap().value();
                                set_draft.update(|d| d.bio = val);
                            }
                        />
                    </div>
                    <div class="form-group">
                        <label for="location">"Location"</label>
                        <input
                            id="location"
                            type="text"
                            class="form-input"
                            placeholder="City, Country"
                            prop:value=move || draft.get().location
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_draft.update(|d| d.location = val);
                            }
                        />
                    </div>
                    <button
                        class="btn-primary"
                        on:click=on_save
                        disabled=move || saving.get()
                    >
                        {move || if saving.get() { "Saving..." } else { "Save Profile" }}
                    </button>
                </div>

                <div class="profile-section">
                    <h3>"Published Credentials"</h3>
                    <p class="text-secondary">"Credentials published from Praxis will appear here with living vitality scores."</p>
                    <p class="text-secondary">"Each credential includes a cryptographic Proof of Learning and Ebbinghaus vitality tracking."</p>
                </div>

                <div class="profile-section">
                    <h3>"Guild Memberships"</h3>
                    <p class="text-secondary">"Your guild roles (Observer → Apprentice → Journeyman → Master → Elder) appear here."</p>
                </div>

                <div class="profile-section">
                    <h3>"Work History"</h3>
                    <p class="text-secondary">"Add your work experience. Peers can verify entries via link attestation."</p>
                    <button class="btn-secondary" disabled>
                        "Add Experience"
                    </button>
                </div>

                <div class="profile-section">
                    <h3>"Endorsements Received"</h3>
                    <p class="text-secondary">"Peer attestations of your skills — no central gatekeeper."</p>
                </div>
            </div>
        </div>
    }
}
