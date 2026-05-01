// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hands-Free Substrate — Local-WASM Voice Commands for Workshop Safety.

use leptos::prelude::*;

#[component]
pub fn VoiceCommandCenter() -> impl IntoView {
    let (is_listening, set_is_listening) = signal(false);
    let (last_command, set_last_command) = signal("None".to_string());

    let toggle_listen = move |_| {
        set_is_listening.update(|v| *v = !*v);
        if is_listening.get() {
            // SIMULATED: Local-WASM Whisper Handshake
            wasm_bindgen_futures::spawn_local(async move {
                gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
                set_last_command.set("Log Temperature: 650\u{00B0}C".to_string());
                set_is_listening.set(false);
            });
        }
    };

    view! {
        <div class="voice-ops-ui" style="margin-top: 1rem; padding: 1rem; background: var(--surface-high); border: 1px solid var(--primary-low); border-radius: 8px">
            <div style="display: flex; justify-content: space-between; align-items: center">
                <div style="font-size: 0.8rem">
                    <strong>"Hands-Free Ops"</strong>
                    <div style="color: var(--text-tertiary); font-size: 0.7rem">
                        {move || if is_listening.get() { "Listening (Local-WASM)..." } else { format!("Last: {}", last_command.get()) }}
                    </div>
                </div>
                <button 
                    class=move || if is_listening.get() { "btn-sm btn-primary pulse" } else { "btn-sm btn-outline" }
                    on:click=toggle_listen
                >
                    {move || if is_listening.get() { "\u{1F399} STOP" } else { "\u{1F399} VOICE" }}
                </button>
            </div>
        </div>
    }
}
