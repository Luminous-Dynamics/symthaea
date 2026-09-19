// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Library view for every ♥ keeper, with the actual audio that was heard
//! (not a recomposed guess). Ported from `loadLiked()` in the legacy
//! `studio/index.html`'s Liked tab: fetches `/api/keepers` once on mount
//! (plus a manual Refresh, since a keep made from Listen while this page
//! isn't mounted wouldn't otherwise show up), then renders one card per
//! saved artifact with MIDI/WAV/Recipe downloads.
//!
//! Kept artifacts remain independent of `MuseState::current`. Audition uses
//! the one persistent shared transport. Newer keeper rows can additionally
//! resolve their stored genealogy manifest on demand; only a successfully
//! cross-checked manifest may contribute a rendition identity to playback.

use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;

use crate::api;
use crate::keeper_api::{self, KeeperRecord};
use crate::state::MuseState;

#[component]
pub fn LikedPage() -> impl IntoView {
    let muse = use_context::<MuseState>().expect("MuseState provided by App");
    let entries = RwSignal::new(Vec::<KeeperRecord>::new());
    let status = RwSignal::new(String::new());
    let identity_notice = RwSignal::new(None::<String>);
    let loaded = RwSignal::new(false);

    let reload = move || {
        status.set("loading…".to_string());
        identity_notice.set(None);
        spawn_local(async move {
            match keeper_api::fetch_keeper_records(api::DEFAULT_BACKEND).await {
                Ok(v) => {
                    entries.set(v);
                    status.set(String::new());
                }
                Err(e) => {
                    entries.set(Vec::new());
                    status.set(format!("couldn't reach the keeper log — {e}"));
                }
            }
            loaded.set(true);
        });
    };

    // Fetch once on mount; `loaded` (rather than a tracked signal) keeps
    // this from re-firing on every unrelated reactive update.
    Effect::new(move |_| {
        if !loaded.get_untracked() {
            reload();
        }
    });

    view! {
        <div class="panel">
            <div class="liked-header">
                <h2>"Library"</h2>
                <p class="muted small">
                    "Every ♥ keeper, with the actual audio you heard — not a recomposed guess. Auditions use the persistent player, so Return restores whatever you were hearing before. Newer keepers verify their stored genealogy before the player is allowed to carry a rendition identity; legacy keepers remain valid identity-less auditions."
                </p>
                <A href="/library/import" attr:class="link-btn">"Import music"</A>
                <button type="button" on:click=move |_| reload()>"Refresh"</button>
            </div>

            {move || identity_notice.get().map(|notice| view! {
                <p class="status-line" role="status">{notice}</p>
            })}

            {move || {
                if !status.get().is_empty() {
                    view! { <p class="muted">{status.get()}</p> }.into_any()
                } else if entries.get().is_empty() {
                    view! {
                        <p class="muted">
                            "Nothing kept yet — ♥ a piece from Listen or Create and it shows up here."
                        </p>
                    }.into_any()
                } else {
                    view! {
                        <div class="candidate-grid">
                            {entries.get().into_iter().map(|record| {
                                let identity_linked = record.genealogy_id.is_some()
                                    && record.score_sha256.is_some();
                                let audition_record = record.clone();
                                let e = record.entry;
                                let when = if e.ts > 0 {
                                    format_timestamp(e.ts)
                                } else {
                                    String::new()
                                };
                                let heading = e.title.clone().unwrap_or_else(|| e.spec.clone());
                                let audition_title = heading.clone();
                                let audition_url =
                                    api::keeper_audio_url(api::DEFAULT_BACKEND, &e.audio_key);
                                let meta = format!(
                                    "{}{} · seed {}",
                                    e.spec,
                                    e.mode.as_ref().map(|m| format!(" · {m}")).unwrap_or_default(),
                                    e.seed,
                                );
                                let grammar_line = format!(
                                    "{}{} · Φ {:.2}",
                                    e.grammar,
                                    e.ending.as_ref().map(|s| format!(" ({s})")).unwrap_or_default(),
                                    e.phi,
                                );
                                view! {
                                    <div class="candidate-card">
                                        <h3>{heading}</h3>
                                        <p class="muted">{meta}</p>
                                        <p class="muted">{grammar_line}</p>
                                        {identity_linked.then(|| view! {
                                            <p class="muted small" title="This keeper row carries both a stored genealogy id and score SHA-256 handle; audition verifies the manifest before promoting rendition identity.">
                                                "identity-linked"
                                            </p>
                                        })}
                                        {(!when.is_empty()).then(|| view! { <p class="muted small">{when}</p> })}
                                        <div class="candidate-actions">
                                            <button
                                                type="button"
                                                aria-label=format!("Audition {audition_title}")
                                                on:click=move |_| {
                                                    let record = audition_record.clone();
                                                    let audio_url = audition_url.clone();
                                                    let title = audition_title.clone();
                                                    identity_notice.set(
                                                        identity_linked.then(|| "verifying saved identity…".to_string())
                                                    );
                                                    spawn_local(async move {
                                                        match keeper_api::fetch_verified_keeper_identity(
                                                            api::DEFAULT_BACKEND,
                                                            &record,
                                                        )
                                                        .await
                                                        {
                                                            Ok(Some(verified)) => {
                                                                identity_notice.set(None);
                                                                muse.play_identity_bound_review_audio(
                                                                    audio_url,
                                                                    title,
                                                                    verified.artifact.rendition,
                                                                );
                                                            }
                                                            Ok(None) => {
                                                                identity_notice.set(None);
                                                                muse.play_review_audio(audio_url, title);
                                                            }
                                                            Err(error) => {
                                                                identity_notice.set(Some(format!(
                                                                    "Audition blocked: saved keeper identity could not be verified — {error}"
                                                                )));
                                                            }
                                                        }
                                                    });
                                                }
                                            >
                                                "Audition"
                                            </button>
                                            {e.midi_available.then(|| view! {
                                                <a class="link-btn"
                                                    href=api::keeper_midi_url(api::DEFAULT_BACKEND, &e.audio_key)
                                                    download=format!("muse_{}.mid", e.audio_key)>
                                                    "MIDI"
                                                </a>
                                            })}
                                            <a class="link-btn"
                                                href=api::keeper_audio_url(api::DEFAULT_BACKEND, &e.audio_key)
                                                download=format!("muse_{}.wav", e.audio_key)>
                                                "WAV"
                                            </a>
                                            <a class="link-btn"
                                                href=api::keeper_recipe_url(api::DEFAULT_BACKEND, &e.audio_key)
                                                download=format!("muse_{}_recipe.json", e.audio_key)>
                                                "Recipe"
                                            </a>
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

/// `e.ts` is Unix seconds. `js_sys::Date` (not `std::time`, which panics on
/// wasm32 without an epoch source) — mirrors the legacy page's
/// `new Date(e.ts * 1000).toLocaleString()`.
fn format_timestamp(ts: u64) -> String {
    let date = js_sys::Date::new_0();
    date.set_time(ts as f64 * 1000.0);
    date.to_locale_string("default", &js_sys::Object::new())
        .into()
}
