// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PWA install prompt — captures beforeinstallprompt and shows native install UI.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;

#[component]
pub fn PwaInstallPrompt() -> impl IntoView {
    let can_install = RwSignal::new(false);
    let dismissed = RwSignal::new(
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .and_then(|s| s.get_item("mycelix_pwa_dismissed").ok().flatten())
            .is_some()
    );

    // Capture beforeinstallprompt event
    Effect::new(move |_| {
        let _ = js_sys::eval("
            window.__pwa_deferred_prompt = null;
            window.addEventListener('beforeinstallprompt', function(e) {
                e.preventDefault();
                window.__pwa_deferred_prompt = e;
                window.__pwa_can_install = true;
            });
            // Check if already installed
            window.addEventListener('appinstalled', function() {
                window.__pwa_can_install = false;
                window.__pwa_deferred_prompt = null;
            });
        ");
    });

    // Poll for install availability (JS events can't directly set Leptos signals)
    let check_install = move || {
        if dismissed.get_untracked() { return; }
        let available = web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w, &JsValue::from_str("__pwa_can_install")).ok())
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if available != can_install.get_untracked() {
            can_install.set(available);
        }
    };

    // Check periodically
    Effect::new(move |_| {
        check_install();
    });

    let on_install = move |_| {
        let _ = js_sys::eval("
            if (window.__pwa_deferred_prompt) {
                window.__pwa_deferred_prompt.prompt();
                window.__pwa_deferred_prompt = null;
                window.__pwa_can_install = false;
            }
        ");
        can_install.set(false);
    };

    let on_dismiss = move |_| {
        dismissed.set(true);
        can_install.set(false);
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten()) {
            let _ = storage.set_item("mycelix_pwa_dismissed", "1");
        }
    };

    view! {
        {move || {
            if !can_install.get() || dismissed.get() { return None; }
            Some(view! {
                <div class="pwa-install-banner">
                    <span class="pwa-install-icon">"📱"</span>
                    <div class="pwa-install-text">
                        <strong>"Install Mycelix Pulse"</strong>
                        <span>" — faster access, offline support"</span>
                    </div>
                    <button class="btn btn-small btn-primary" on:click=on_install>"Install"</button>
                    <button class="btn btn-small btn-ghost pwa-dismiss" on:click=on_dismiss>"✕"</button>
                </div>
            })
        }}
    }
}
