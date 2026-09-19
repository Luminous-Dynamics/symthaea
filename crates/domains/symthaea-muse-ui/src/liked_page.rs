// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Library view for every ♥ keeper, with the actual audio that was heard
//! (not a recomposed guess). Ported from `loadLiked()` in the legacy
//! `studio/index.html`'s Liked tab: fetches `/api/keepers` once on mount
//! (plus a manual Refresh, since a keep made from Listen while this page
//! isn't mounted wouldn't otherwise show up), then renders one card per
//! entry with its own saved `/api/keeper-audio/{audio_key}` artifact plus
//! MIDI/WAV/Recipe downloads.
//!
//! Kept artifacts remain independent of `MuseState::current`. A later
//! audition-foundation tranche will move keeper playback onto the shared
//! persistent transport without changing that identity boundary.

use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;

use crate::api::{self, KeeperEntry};

#[component]
pub fn LikedPage() -> impl IntoView {
    let entries = RwSignal::new(Vec::<KeeperEntry>::new());
    let status = RwSignal::new(String::new());
    let loaded = RwSignal::new(false);

    let reload = move || {
        status.set("loading…".to_string());
        spawn_local(async move {
            match api::fetch_keepers(api::DEFAULT_BACKEND).await {
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
                    "Every ♥ keeper, with the actual audio you heard — not a recomposed guess. Private symbolic import is available separately while unified library indexing remains a later tranche."
                </p>
                <A href="/library/import" attr:class="link-btn">"Import music"</A>
                <button type="button" on:click=move |_| reload()>"Refresh"</button>
            </div>

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
                            {entries.get().into_iter().map(|e| {
                                let when = if e.ts > 0 {
                                    format_timestamp(e.ts)
                                } else {
                                    String::new()
                                };
                                let heading = e.title.clone().unwrap_or_else(|| e.spec.clone());
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
                                        {(!when.is_empty()).then(|| view! { <p class="muted small">{when}</p> })}
                                        <audio controls preload="none"
                                            src=api::keeper_audio_url(api::DEFAULT_BACKEND, &e.audio_key)>
                                        </audio>
                                        <div class="candidate-actions">
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
