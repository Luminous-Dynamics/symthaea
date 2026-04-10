// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// NOTE: Craft zome calls (craft_graph.*) require the unified hApp with a "craft" role.
// Currently calls use the Praxis default role — these will need cross-role dispatch
// once the Praxis frontend migrates to mycelix-leptos-core's HolochainProviderAuto.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::holochain::{use_holochain, ConnectionBadge, HolochainProvider};

use crate::pages::credentials::real_credentials;
use crate::persistence;
use crate::student_profile::use_profile;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct ProfessionalProfileDraft {
    display_name: String,
    headline: String,
    bio: String,
    location: String,
    website: String,
    avatar_url: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct PublishedState {
    published_ids: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProfessionalProfileInput {
    display_name: String,
    headline: String,
    bio: String,
    location: Option<String>,
    website: Option<String>,
    avatar_url: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PublishedCredentialInput {
    credential_id: String,
    title: String,
    issuer: String,
    issued_on: String,
    expires_on: Option<String>,
    source_dna: Option<String>,
    entry_hash: Option<String>,
    action_hash: Option<String>,
    summary: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SkillEndorsementInput {
    endorsed_agent: String,
    skill: String,
    rationale: String,
    evidence: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SkillEndorsementView {
    endorsed_agent: String,
    skill: String,
    rationale: String,
    evidence: Option<String>,
    created_at: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct EndorsementDraft {
    endorsed_agent: String,
    skill: String,
    rationale: String,
    evidence: String,
}

const PROFILE_DRAFT_KEY: &str = "praxis_professional_profile_draft";
const PROFILE_PUBLISH_KEY: &str = "praxis_professional_published";
const ENDORSE_DRAFT_KEY: &str = "praxis_professional_endorsement_draft";
const PROFILE_VIEW_KEY: &str = "praxis_professional_view_agent";
const PROFILE_FILTER_KEY: &str = "praxis_professional_filter_skill";

fn initial_profile(student_name: String) -> ProfessionalProfileDraft {
    let mut draft = persistence::load::<ProfessionalProfileDraft>(PROFILE_DRAFT_KEY).unwrap_or_default();
    if draft.display_name.is_empty() && !student_name.is_empty() {
        draft.display_name = student_name;
    }
    draft
}

#[component]
pub fn ProfilePage() -> impl IntoView {
    // Profile page connects to the Professional DNA for credential publishing.
    // In mock mode, all operations use localStorage.
    view! {
        <HolochainProvider>
            <ProfileInner />
        </HolochainProvider>
    }
}

#[component]
fn ProfileInner() -> impl IntoView {
    let student = use_profile();
    let hc = use_holochain();
    let hc_save = hc.clone();
    let hc_endorsements = hc.clone();
    let hc_pubkey = hc.clone();
    let hc_endorse = hc.clone();

    let (draft, set_draft) = signal(initial_profile(student.get().name));
    let (publish_state, set_publish_state) =
        signal(persistence::load::<PublishedState>(PROFILE_PUBLISH_KEY).unwrap_or_default());
    let (endorse_draft, set_endorse_draft) =
        signal(persistence::load::<EndorsementDraft>(ENDORSE_DRAFT_KEY).unwrap_or_default());
    let (view_agent, set_view_agent) =
        signal(persistence::load::<String>(PROFILE_VIEW_KEY).unwrap_or_default());
    let (filter_skill, set_filter_skill) =
        signal(persistence::load::<String>(PROFILE_FILTER_KEY).unwrap_or_default());
    let (status, set_status) = signal::<Option<String>>(None);
    let (copied_key, set_copied_key) = signal(false);

    Effect::new(move |_| {
        persistence::save(PROFILE_DRAFT_KEY, &draft.get());
    });

    Effect::new(move |_| {
        persistence::save(PROFILE_PUBLISH_KEY, &publish_state.get());
    });

    Effect::new(move |_| {
        persistence::save(ENDORSE_DRAFT_KEY, &endorse_draft.get());
    });

    Effect::new(move |_| {
        persistence::save(PROFILE_VIEW_KEY, &view_agent.get());
    });

    Effect::new(move |_| {
        persistence::save(PROFILE_FILTER_KEY, &filter_skill.get());
    });

    let credentials = real_credentials();
    let my_pubkey = LocalResource::new(move || {
        let hc = hc_pubkey.clone();
        async move {
            match hc
                .call_zome_default::<(), String>(
                    "craft_graph",
                    "get_my_agent_pubkey",
                    &(),
                )
                .await
            {
                Ok(key) => key,
                Err(_) => String::new(),
            }
        }
    });
    let endorsements = LocalResource::new(move || {
        let hc = hc_endorsements.clone();
        async move {
            let target = view_agent.get();
            let result = if target.trim().is_empty() {
                hc.call_zome_default::<(), Vec<SkillEndorsementView>>(
                    "craft_graph",
                    "get_my_endorsements_view",
                    &(),
                )
                .await
            } else {
                hc.call_zome_default::<String, Vec<SkillEndorsementView>>(
                    "craft_graph",
                    "list_skill_endorsements_view",
                    &target,
                )
                .await
            };

            match result {
                Ok(list) => list,
                Err(_) => Vec::new(),
            }
        }
    });

    let on_save = move |_| {
        let hc = hc_save.clone();
        let payload = ProfessionalProfileInput {
            display_name: draft.get().display_name,
            headline: draft.get().headline,
            bio: draft.get().bio,
            location: if draft.get().location.is_empty() { None } else { Some(draft.get().location) },
            website: if draft.get().website.is_empty() { None } else { Some(draft.get().website) },
            avatar_url: if draft.get().avatar_url.is_empty() { None } else { Some(draft.get().avatar_url) },
        };

        if hc.is_mock() {
            set_status.set(Some("Saved locally — connect a conductor to publish.".into()));
            return;
        }

        spawn_local(async move {
            let result: Result<(), String> = hc
                .call_zome_default("craft_graph", "set_profile", &payload)
                .await;
            match result {
                Ok(_) => set_status.set(Some("Profile published to professional graph.".into())),
                Err(e) => set_status.set(Some(format!("Publish failed: {e}"))),
            }
        });
    };

    let on_endorse = move |_| {
        let hc = hc_endorse.clone();
        let draft = endorse_draft.get();
        let set_endorse_draft = set_endorse_draft.clone();
        let mut endorsed_agent = draft.endorsed_agent.trim().to_string();
        if endorsed_agent.is_empty() {
            let viewed = view_agent.get();
            if !viewed.trim().is_empty() {
                endorsed_agent = viewed.trim().to_string();
            }
        }
        if endorsed_agent.is_empty() || draft.skill.trim().is_empty() || draft.rationale.trim().is_empty() {
            set_status.set(Some("Provide agent pubkey, skill, and rationale.".into()));
            return;
        }

        let payload = SkillEndorsementInput {
            endorsed_agent,
            skill: draft.skill.trim().to_string(),
            rationale: draft.rationale.trim().to_string(),
            evidence: if draft.evidence.trim().is_empty() { None } else { Some(draft.evidence.trim().to_string()) },
        };

        if hc.is_mock() {
            set_status.set(Some("Endorsement saved locally — connect to publish.".into()));
            return;
        }

        spawn_local(async move {
            let result: Result<(), String> = hc
                .call_zome_default("craft_graph", "endorse_skill", &payload)
                .await;
            match result {
                Ok(_) => {
                    set_status.set(Some("Endorsement published to professional graph.".into()));
                    set_endorse_draft.set(EndorsementDraft::default());
                }
                Err(e) => set_status.set(Some(format!("Endorsement failed: {e}"))),
            }
        });
    };

    view! {
        <div class="profile-page">
            <a href="/stewardship" class="refuge-back-link">"\u{2190} Stewardship"</a>
            <div class="profile-header">
                <div>
                    <h1>"Stewardship Portfolio"</h1>
                    <p>
                        "Opt-in, verifiable professional identity. Publish only the credentials you choose."
                    </p>
                </div>
                <div class="profile-connection">
                    <ConnectionBadge />
                </div>
            </div>

            {move || {
                status.get().map(|s| view! { <div class="profile-status">{s}</div> })
            }}

            <section class="profile-card profile-identity">
                <div class="profile-section-header">
                    <div>
                        <h2>"Identity Key"</h2>
                        <p>"Share this AgentPubKey to receive endorsements."</p>
                    </div>
                    <span class="profile-pill">"Base64"</span>
                </div>
                <div class="profile-identity-row">
                    <Suspense fallback=move || view! { <div class="profile-empty">"Connecting..."</div> }>
                        {move || my_pubkey.get().map(|key| {
                            if key.is_empty() {
                                view! { <div class="profile-empty">"Connect to fetch your key."</div> }.into_any()
                            } else {
                                view! {
                                    <>
                                        <input class="profile-identity-input" type="text" readonly value=key.clone() />
                                        <button
                                            class="profile-copy"
                                            aria-label="Copy agent public key to clipboard"
                                            on:click=move |_| {
                                                if let Some(window) = web_sys::window() {
                                                    let clipboard = window.navigator().clipboard();
                                                    let _ = clipboard.write_text(&key);
                                                    set_copied_key.set(true);
                                                    wasm_bindgen_futures::spawn_local(async move {
                                                        gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
                                                        set_copied_key.set(false);
                                                    });
                                                }
                                            }
                                        >
                                            {move || if copied_key.get() { "\u{2714} Copied" } else { "\u{1F4CB} Copy" }}
                                        </button>
                                    </>
                                }.into_any()
                            }
                        })}
                    </Suspense>
                </div>
            </section>

            <section class="profile-card">
                <h2>"Profile"</h2>
                <div class="profile-grid">
                    <label class="profile-field">
                        <span>"Display name"</span>
                        <input
                            type="text"
                            prop:value=move || draft.get().display_name
                            on:input=move |ev| {
                                set_draft.update(|d| d.display_name = event_target_value(&ev));
                            }
                        />
                    </label>
                    <label class="profile-field">
                        <span>"Headline"</span>
                        <input
                            type="text"
                            prop:value=move || draft.get().headline
                            on:input=move |ev| {
                                set_draft.update(|d| d.headline = event_target_value(&ev));
                            }
                        />
                    </label>
                    <label class="profile-field">
                        <span>"Location"</span>
                        <input
                            type="text"
                            prop:value=move || draft.get().location
                            on:input=move |ev| {
                                set_draft.update(|d| d.location = event_target_value(&ev));
                            }
                        />
                    </label>
                    <label class="profile-field">
                        <span>"Website"</span>
                        <input
                            type="url"
                            prop:value=move || draft.get().website
                            on:input=move |ev| {
                                set_draft.update(|d| d.website = event_target_value(&ev));
                            }
                        />
                    </label>
                    <label class="profile-field profile-field-wide">
                        <span>"Bio"</span>
                        <textarea
                            prop:value=move || draft.get().bio
                            on:input=move |ev| {
                                set_draft.update(|d| d.bio = event_target_value(&ev));
                            }
                        />
                    </label>
                </div>
                <div class="profile-actions">
                    <button class="profile-save" on:click=on_save>"Publish Profile"</button>
                    <span class="profile-note">
                        "Defaults to private until you opt-in to publish."
                    </span>
                </div>
            </section>

            <section class="profile-card">
                <div class="profile-section-header">
                    <div>
                        <h2>"Verified Credentials"</h2>
                        <p>"Select credentials to publish into the professional graph."</p>
                    </div>
                    <span class="profile-pill">"Opt-in per credential"</span>
                </div>

                <div class="profile-credentials">
                    {credentials.into_iter().map(|cred| {
                        let credential_id = cred.credential_id.clone();
                        let credential_id_for_state = credential_id.clone();
                        let course_name = cred.course_name.clone();
                        let issuer = cred.issuer.clone();
                        let issuance_date = cred.issuance_date.clone();
                        let expiration_date = cred.expiration_date.clone();
                        let score_band = cred.score_band.clone();
                        let payload_course_name = course_name.clone();
                        let payload_issuer = issuer.clone();
                        let payload_issuance_date = issuance_date.clone();
                        let payload_expiration_date = expiration_date.clone();
                        let payload_score_band = score_band.clone();
                        let hc = hc.clone();
                        let set_status = set_status.clone();
                        let publish_state = publish_state.clone();
                        let set_publish_state = set_publish_state.clone();
                        let published = Memo::new(move |_| {
                            publish_state
                                .get()
                                .published_ids
                                .contains(&credential_id_for_state)
                        });

                        let on_toggle = move |_| {
                            let hc = hc.clone();
                            let mut next = publish_state.get();
                            if next.published_ids.contains(&credential_id) {
                                next.published_ids.retain(|id| id != &credential_id);
                                set_status.set(Some("Credential hidden from professional graph.".into()));
                                set_publish_state.set(next);
                                return;
                            }

                            next.published_ids.push(credential_id.clone());
                            set_publish_state.set(next);

                            if hc.is_mock() {
                                set_status.set(Some("Saved locally — connect to publish.".into()));
                                return;
                            }

                            let payload = PublishedCredentialInput {
                                credential_id: credential_id.clone(),
                                title: payload_course_name.clone(),
                                issuer: payload_issuer.clone(),
                                issued_on: payload_issuance_date.clone(),
                                expires_on: payload_expiration_date.clone(),
                                source_dna: Some("praxis".into()),
                                entry_hash: None,
                                action_hash: None,
                                summary: Some(format!("Score band: {}", payload_score_band)),
                            };

                            spawn_local(async move {
                                let result: Result<(), String> = hc
                                    .call_zome_default("craft_graph", "publish_credential", &payload)
                                    .await;
                                if let Err(e) = result {
                                    set_status.set(Some(format!("Publish failed: {e}")));
                                } else {
                                    set_status.set(Some("Credential published to professional graph.".into()));
                                }
                            });
                        };

                        view! {
                            <div class="profile-credential">
                                <div class="profile-credential-main">
                                    <h3>{course_name.clone()}</h3>
                                    <p>{issuer.clone()}</p>
                                    <span class="profile-credential-meta">
                                        {issuance_date.clone()}
                                    </span>
                                </div>
                                <button
                                    class=move || if published.get() { "publish-toggle active" } else { "publish-toggle" }
                                    on:click=on_toggle
                                >
                                    {move || if published.get() { "Published" } else { "Publish" }}
                                </button>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>

            <section class="profile-card profile-cta">
                <h2>"Professional Graph"</h2>
                <p>
                    "Your published credentials are indexed in the professional DNA and can be"
                    " shared as verifiable references across the Mycelix mesh."
                </p>
                <a class="profile-link" href="/governance">"Review governance policies \u{2192}"</a>
            </section>

            <section class="profile-card">
                <div class="profile-section-header">
                    <div>
                        <h2>"Professional Vouching"</h2>
                        <p>"Peer attestations anchored in the Web-of-Trust."</p>
                    </div>
                    <span class="profile-pill profile-pill-trust">"Attestation"</span>
                </div>

                <div class="endorsement-controls">
                    <label class="profile-field profile-field-wide">
                        <span>"View AgentPubKey (leave blank for self)"</span>
                        <input
                            type="text"
                            prop:value=move || view_agent.get()
                            on:input=move |ev| set_view_agent.set(event_target_value(&ev))
                        />
                    </label>
                    <label class="profile-field">
                        <span>"Filter by skill"</span>
                        <input
                            type="text"
                            prop:value=move || filter_skill.get()
                            on:input=move |ev| set_filter_skill.set(event_target_value(&ev))
                        />
                    </label>
                    <button class="publish-toggle" on:click=move |_| set_view_agent.set(String::new())>
                        "View Self"
                    </button>
                    <button class="publish-toggle" on:click=move |_| set_filter_skill.set(String::new())>
                        "Clear Filter"
                    </button>
                    <button
                        class="publish-toggle"
                        on:click=move |_| {
                            let target = view_agent.get();
                            if !target.trim().is_empty() {
                                set_endorse_draft.update(|d| d.endorsed_agent = target);
                            }
                        }
                    >
                        "Use Viewed Agent"
                    </button>
                </div>
                {move || {
                    let active = filter_skill.get();
                    if active.trim().is_empty() {
                        view! { <div></div> }.into_any()
                    } else {
                        view! {
                            <div class="filter-chip">
                                <span>"Filter: " {active.clone()}</span>
                                <button on:click=move |_| set_filter_skill.set(String::new())>"Clear"</button>
                            </div>
                        }.into_any()
                    }
                }}

                <Suspense fallback=move || view! { <div class="profile-empty">"Loading endorsements..."</div> }>
                    {move || endorsements.get().map(|list| {
                        let needle = filter_skill.get().trim().to_lowercase();
                        let filtered: Vec<SkillEndorsementView> = if needle.is_empty() {
                            list
                        } else {
                            list.into_iter()
                                .filter(|endorsement| endorsement.skill.to_lowercase().contains(&needle))
                                .collect()
                        };

                        if filtered.is_empty() {
                            if needle.is_empty() {
                                view! { <div class="profile-empty">"No endorsements received yet."</div> }.into_any()
                            } else {
                                view! { <div class="profile-empty">"No endorsements match that skill filter."</div> }.into_any()
                            }
                        } else {
                            view! {
                                <div class="endorsement-grid">
                                    {filtered.into_iter().map(|endorsement| {
                                        let evidence = endorsement.evidence.clone();
                                        view! {
                                            <div class="endorsement-card">
                                                <div class="endorsement-header">
                                                    <span class="endorsement-skill">{endorsement.skill}</span>
                                                    <span class="endorsement-badge">"Vouched"</span>
                                                </div>
                                                <p class="endorsement-rationale">{endorsement.rationale}</p>
                                                <div class="endorsement-meta">
                                                    <span class="endorsement-agent">
                                                        "Endorsed agent: " {endorsement.endorsed_agent}
                                                    </span>
                                                    <span class="endorsement-date">{endorsement.created_at}</span>
                                                </div>
                                                {evidence.map(|url| view! {
                                                    <a class="endorsement-evidence" href=url target="_blank">"Evidence \u{2192}"</a>
                                                })}
                                            </div>
                                        }
                                    }).collect_view()}
                                </div>
                            }.into_any()
                        }
                    })}
                </Suspense>

                <div class="endorsement-form">
                    <h3>"Issue an Endorsement"</h3>
                    <div class="profile-grid">
                        <label class="profile-field profile-field-wide">
                            <span>"AgentPubKey (base64)"</span>
                            <input
                                type="text"
                                prop:value=move || endorse_draft.get().endorsed_agent
                                on:input=move |ev| {
                                    set_endorse_draft.update(|d| d.endorsed_agent = event_target_value(&ev));
                                }
                            />
                        </label>
                        <label class="profile-field">
                            <span>"Skill"</span>
                            <input
                                type="text"
                                prop:value=move || endorse_draft.get().skill
                                on:input=move |ev| {
                                    set_endorse_draft.update(|d| d.skill = event_target_value(&ev));
                                }
                            />
                        </label>
                        <label class="profile-field">
                            <span>"Evidence URL (optional)"</span>
                            <input
                                type="url"
                                prop:value=move || endorse_draft.get().evidence
                                on:input=move |ev| {
                                    set_endorse_draft.update(|d| d.evidence = event_target_value(&ev));
                                }
                            />
                        </label>
                        <label class="profile-field profile-field-wide">
                            <span>"Rationale"</span>
                            <textarea
                                prop:value=move || endorse_draft.get().rationale
                                on:input=move |ev| {
                                    set_endorse_draft.update(|d| d.rationale = event_target_value(&ev));
                                }
                            />
                        </label>
                    </div>
                    <div class="profile-actions">
                        <button class="profile-save profile-save-endorse" on:click=on_endorse>
                            "Publish Endorsement"
                        </button>
                        <span class="profile-note">
                            "This is a peer attestation, not a like."
                        </span>
                    </div>
                </div>
            </section>
        </div>
    }
}
