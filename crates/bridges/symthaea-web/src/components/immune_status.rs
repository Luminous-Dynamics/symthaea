// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::state::AppState;
use crate::worker::EngineWorker;

/// Small badge showing the immune system safety level (Green/Yellow/Orange/Red).
/// Polls `safetyLevel` from the worker periodically.
#[component]
pub fn ImmuneStatus() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    // Poll safety level every ~3 seconds
    let state_poll = state.clone();
    let engine_poll = engine.clone();
    Effect::new(move |_| {
        if !state_poll.worker_ready.get() {
            return;
        }
        let engine = engine_poll.clone();
        let state = state_poll.clone();
        wasm_bindgen_futures::spawn_local(async move {
            loop {
                let promise = engine.send_simple("safetyLevel");
                if let Ok(result) = JsFuture::from(promise).await {
                    if let Some(level) = result.as_string() {
                        state.safety_level.set(level);
                    } else if let Ok(level) = js_sys::Reflect::get(&result, &"level".into()) {
                        if let Some(s) = level.as_string() {
                            state.safety_level.set(s);
                        }
                    }
                }
                gloo_timers::future::TimeoutFuture::new(3000).await;
            }
        });
    });

    let dot_color = move || match state.safety_level.get().as_str() {
        "Red" => "#ef4444",
        "Orange" => "#f59e0b",
        "Yellow" => "#eab308",
        _ => "var(--leaf-green)",
    };

    view! {
        <div class="immune-badge" title=move || format!("Safety: {}", state.safety_level.get())>
            <span class="immune-dot" style:background=dot_color />
            <span class="immune-label">{move || state.safety_level.get()}</span>
        </div>
    }
}
