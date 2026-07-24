// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research Mode's Harmony view: the shared current piece's key, and the
//! 7 diatonic triads that key offers (Roman numeral, chord symbol,
//! harmonic function) — from `GET /api/harmony/{id}`.
//!
//! Deliberately NOT a chord-by-time analysis of the piece itself (which
//! chords it actually used, when) — that data is computed mid-compose by
//! the engine and discarded before reaching the client; see
//! `harmony_summary()`'s doc comment in `muse_studio.rs`. This is the
//! smaller, honest step: what harmonic vocabulary does this key offer.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, HarmonySummary};
use crate::state::MuseState;

fn function_class(function: &str) -> &'static str {
    match function {
        "Predominant" => "harmony-predominant",
        "Dominant" => "harmony-dominant",
        _ => "harmony-tonic",
    }
}

#[component]
pub fn HarmonyView(muse: MuseState) -> impl IntoView {
    let summary = RwSignal::new(None::<HarmonySummary>);
    let load_status = RwSignal::new(String::new());

    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            summary.set(None);
            return;
        };
        let id = c.id;
        load_status.set("loading harmony…".to_string());
        spawn_local(async move {
            match api::fetch_harmony(api::DEFAULT_BACKEND, id).await {
                Ok(s) => {
                    summary.set(Some(s));
                    load_status.set(String::new());
                }
                Err(e) => load_status.set(format!("harmony unavailable: {e}")),
            }
        });
    });

    view! {
        <div class="harmony-view">
            {move || match summary.get() {
                None => view! { <p class="muted small">{move || load_status.get()}</p> }.into_any(),
                Some(s) => view! {
                    <p class="muted">
                        {format!("Key: {} {}", s.tonic, s.tonality)}
                    </p>
                    <div class="harmony-chords">
                        {s.chords.into_iter().map(|c| {
                            let class = format!("harmony-chord {}", function_class(&c.function));
                            view! {
                                <div class=class>
                                    <span class="harmony-roman">{c.roman}</span>
                                    <span class="harmony-symbol">{c.symbol}</span>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                    <p class="legend">
                        <span class="legend-item"><b class="harmony-tonic">"■"</b>" Tonic"</span>
                        <span class="legend-item"><b class="harmony-predominant">"■"</b>" Predominant"</span>
                        <span class="legend-item"><b class="harmony-dominant">"■"</b>" Dominant"</span>
                    </p>
                }.into_any(),
            }}
        </div>
    }
}
