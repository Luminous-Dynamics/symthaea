// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Notification permission soft-prompt — asks once, remembers dismissal.

use leptos::prelude::*;

const DISMISSED_KEY: &str = "mycelix_pulse_notif_dismissed";

#[component]
pub fn NotificationPrompt() -> impl IntoView {
    let already_dismissed = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(DISMISSED_KEY).ok().flatten())
        .is_some();

    let show = RwSignal::new(!already_dismissed);

    // Check if already granted
    let granted = js_sys::eval("typeof Notification !== 'undefined' && Notification.permission === 'granted'")
        .ok().and_then(|v| v.as_bool()).unwrap_or(false);
    if granted { show.set(false); }

    let denied = js_sys::eval("typeof Notification !== 'undefined' && Notification.permission === 'denied'")
        .ok().and_then(|v| v.as_bool()).unwrap_or(false);
    if denied { show.set(false); }

    let dismiss = move || {
        show.set(false);
        if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = s.set_item(DISMISSED_KEY, "1");
        }
    };

    let on_enable = move |_| {
        crate::notifications::request_notification_permission();
        dismiss();
    };
    let on_dismiss = move |_| { dismiss(); };

    view! {
        <div class="notif-prompt" style=move || if show.get() { "" } else { "display:none" }>
            <span class="notif-prompt-icon">"\u{1F514}"</span>
            <div class="notif-prompt-text">
                <strong>"Get notified of new mail?"</strong>
                <p>"We'll only notify you for messages, never spam."</p>
            </div>
            <button class="btn btn-sm btn-primary" on:click=on_enable>"Enable"</button>
            <button class="btn btn-sm btn-secondary" on:click=on_dismiss>"Not now"</button>
        </div>
    }
}
