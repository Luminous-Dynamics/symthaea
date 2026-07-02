// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profile page — craft identity, work history, credentials, and endorsements.
//! Mock-first: renders from localStorage, syncs to conductor when connected.

use leptos::prelude::*;
use mycelix_leptos_core::{
    holochain_provider::use_holochain,
    toasts::{use_toasts, ToastKind},
    SovereignRadar, SovereignRadarSize,
};

use crate::context::{use_craft, CraftProfile};
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

impl From<CraftProfile> for ProfileDraft {
    fn from(p: CraftProfile) -> Self {
        Self {
            display_name: p.display_name,
            headline: p.headline,
            bio: p.bio,
            location: p.location.unwrap_or_default(),
            website: p.website.unwrap_or_default(),
        }
    }
}

impl From<&ProfileDraft> for CraftProfile {
    fn from(d: &ProfileDraft) -> Self {
        Self {
            display_name: d.display_name.clone(),
            headline: d.headline.clone(),
            bio: d.bio.clone(),
            location: if d.location.is_empty() { None } else { Some(d.location.clone()) },
            website: if d.website.is_empty() { None } else { Some(d.website.clone()) },
            avatar_url: None,
        }
    }
}

#[component]
pub fn ProfilePage() -> impl IntoView {
    let craft = use_craft();
    let hc = use_holochain();
    let toasts = use_toasts();

    // Initialize from conductor profile or localStorage
    let initial = craft.profile.get_untracked()
        .map(ProfileDraft::from)
        .unwrap_or_else(|| persistence::load::<ProfileDraft>(PROFILE_KEY).unwrap_or_default());
    let (draft, set_draft) = signal(initial);
    let (saving, set_saving) = signal(false);

    // Persist to localStorage on every change
    Effect::new(move |_| {
        persistence::save(PROFILE_KEY, &draft.get());
    });

    // Save handler — tries conductor, falls back to localStorage
    let on_save = move |_| {
        let hc = hc.clone();
        let toasts = toasts.clone();
        let profile: CraftProfile = (&draft.get_untracked()).into();
        set_saving.set(true);

        wasm_bindgen_futures::spawn_local(async move {
            if hc.is_mock() {
                toasts.push("Profile saved locally. Connect conductor to publish to network.", ToastKind::Info);
            } else {
                match hc.call_zome_default::<CraftProfile, ()>("craft_graph", "set_profile", &profile).await {
                    Ok(_) => toasts.success("Profile published to the Holochain network!"),
                    Err(e) => toasts.error(format!("Failed to publish: {e}")),
                }
            }
            set_saving.set(false);
        });
    };

    view! {
        <div class="page profile-page">
            <h1>"Craft Profile"</h1>

            <div class="profile-grid">
                <div class="profile-section">
                    <h3>"Identity"</h3>
                    <div class="form-group">
                        <label for="display_name">"Display Name"</label>
                        <input id="display_name" type="text" class="form-input"
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
                        <input id="headline" type="text" class="form-input"
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
                        <textarea id="bio" rows="4" class="form-textarea"
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
                        <input id="location" type="text" class="form-input"
                            placeholder="City, Country"
                            prop:value=move || draft.get().location
                            on:input=move |ev| {
                                use wasm_bindgen::JsCast;
                                let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                                set_draft.update(|d| d.location = val);
                            }
                        />
                    </div>
                    <button class="btn-primary" on:click=on_save disabled=move || saving.get()>
                        {move || if saving.get() { "Publishing..." } else { "Save & Publish" }}
                    </button>
                </div>

                <div class="profile-section">
                    <h3>"Published Credentials"</h3>
                    {move || {
                        let creds = craft.credentials.get();
                        if creds.is_empty() {
                            view! {
                                <p class="text-secondary">"Import credentials from Praxis to see them here with living vitality scores."</p>
                                <a href="/credentials" class="btn-secondary">"Go to Credentials"</a>
                            }.into_any()
                        } else {
                            view! {
                                <div class="mini-cred-list">
                                    {creds.iter().map(|c| view! {
                                        <div class="mini-cred">
                                            <span>{c.title.clone()}</span>
                                            <span class="vitality-mini">{c.vitality_permille / 10}"%"</span>
                                        </div>
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>

                <div class="profile-section">
                    <crate::components::SkillRadar />
                    <div class="profile-section">
                        <h3>"Civic Profile"</h3>
                        <SovereignRadar size=SovereignRadarSize::Small />
                    </div>
                </div>

                <div class="profile-section">
                    <crate::components::LemHeatmap />
                </div>

                <div class="profile-section">
                    <h3>"Guild Memberships"</h3>
                    {move || {
                        let guilds = craft.guilds.get();
                        if guilds.is_empty() {
                            view! {
                                <p class="text-secondary">"Join a guild to start your apprenticeship journey."</p>
                            }.into_any()
                        } else {
                            view! {
                                <div class="mini-guild-list">
                                    {guilds.iter().map(|g| view! {
                                        <div class="mini-guild">
                                            <span>{g.guild_name.clone()}</span>
                                            <span class="role-badge">{g.role.clone()}</span>
                                        </div>
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>

                <div class="profile-section">
                    <h3>"Work History"</h3>
                    <p class="text-secondary">"Add work experience. Peers verify via link attestation."</p>
                    <button class="btn-secondary" disabled>"Add Experience"</button>
                </div>

                <div class="profile-section">
                    <h3>"Endorsements"</h3>
                    <p class="text-secondary">
                        {move || format!("{} peer attestations", craft.endorsement_count.get())}
                    </p>
                </div>
            </div>
        </div>
    }
}
