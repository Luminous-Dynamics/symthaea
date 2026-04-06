// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profile page — professional identity, work history, credentials, and endorsements.

use leptos::prelude::*;

use crate::persistence;

const PROFILE_KEY: &str = "professional_profile_draft";

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

    // Persist on change
    Effect::new(move |_| {
        persistence::save(PROFILE_KEY, &draft.get());
    });

    view! {
        <div class="page profile-page">
            <h1>"Professional Profile"</h1>

            <div class="profile-grid">
                <div class="profile-section">
                    <h3>"Identity"</h3>
                    <div class="form-group">
                        <label for="display_name">"Display Name"</label>
                        <input
                            id="display_name"
                            type="text"
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
                            placeholder="Tell your professional story..."
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
                            placeholder="City, Country"
                            prop:value=move || draft.get().location
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_draft.update(|d| d.location = val);
                            }
                        />
                    </div>
                    <button class="btn-primary" disabled title="Connect to Holochain conductor to save profile">
                        "Save to Network"
                    </button>
                </div>

                <div class="profile-section">
                    <h3>"Published Credentials"</h3>
                    <p class="text-secondary">"Credentials published from EduNet will appear here."</p>
                    <p class="text-secondary">"Each credential includes a cryptographic Proof of Learning."</p>
                </div>

                <div class="profile-section">
                    <h3>"Work History"</h3>
                    <p class="text-secondary">"Add your work experience. Peers can verify entries."</p>
                    <button class="btn-secondary" disabled title="Connect to conductor to add work history">
                        "Add Experience"
                    </button>
                </div>

                <div class="profile-section">
                    <h3>"Endorsements Received"</h3>
                    <p class="text-secondary">"Peer attestations of your skills."</p>
                </div>
            </div>
        </div>
    }
}
