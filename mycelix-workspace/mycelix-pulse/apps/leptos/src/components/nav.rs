// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;

use crate::holochain::{use_holochain, ConnectionStatus};
use crate::mail_context::use_mail;
use crate::theme::use_theme;
use crate::offline::use_offline;

#[component]
pub fn Nav() -> impl IntoView {
    let hc = use_holochain();
    let mail = use_mail();
    let theme = use_theme();
    let offline = use_offline();

    let status_label = move || match hc.status.get() {
        ConnectionStatus::Connected => "Live",
        ConnectionStatus::Connecting => "Connecting",
        ConnectionStatus::Mock => "Mock",
        ConnectionStatus::Disconnected => "Offline",
    };

    let status_class = move || match hc.status.get() {
        ConnectionStatus::Connected => "status-dot connected",
        ConnectionStatus::Connecting => "status-dot connecting",
        _ => "status-dot mock",
    };

    let unread = move || {
        let count = mail.inbox.get().iter().filter(|e| !e.is_read).count();
        if count > 0 { format!(" ({count})") } else { String::new() }
    };

    let mail_key = mail.clone();
    let key_class = move || mail_key.key_status.get().css_class().to_string();
    let mail_key2 = mail.clone();
    let key_label = move || mail_key2.key_status.get().label();

    // Theme panel toggle
    let show_theme_panel: RwSignal<bool> = expect_context();
    let on_theme_toggle = move |_| {
        show_theme_panel.update(|v| *v = !*v);
    };
    let theme_icon = move || {
        if theme.current.get() == mail_leptos_types::Theme::Dark { "\u{1F319}" } else { "\u{2600}" }
    };

    // Offline queue indicator (#4)
    let queue_count = move || offline.queue_size.get();

    view! {
        <nav class="navbar" aria-label="Primary navigation">
            <a href="/" class="logo">
                <span class="logo-icon">"\u{2709}"</span>
                <span class="logo-text">"Mail"</span>
                <span class="unread-badge">{unread}</span>
            </a>
            <div class="nav-links">
                <A href="/">"Inbox"</A>
                <A href="/compose">"Compose"</A>
                <A href="/contacts">"Contacts"</A>
                <A href="/search">"Search"</A>
                <A href="/accounts">"Accounts"</A>
                <A href="/settings">"Settings"</A>
            </div>
            <crate::components::NotificationCenter />
            // Visible command palette trigger (discoverability)
            <button class="palette-trigger" on:click=move |_| {
                // Simulate Ctrl+K to open palette
                let _ = js_sys::eval("window.dispatchEvent(new KeyboardEvent('keydown',{key:'k',ctrlKey:true,bubbles:true}))");
            } title="Command palette (Ctrl+K)">
                "\u{1F50D}"
            </button>
            <div class="nav-status">
                {move || {
                    let qc = queue_count();
                    (qc > 0).then(|| view! {
                        <span class="offline-badge" title=format!("{qc} queued actions")>
                            {format!("{qc} queued")}
                        </span>
                    })
                }}
                <button class="theme-toggle" on:click=on_theme_toggle title="Toggle theme">
                    {theme_icon}
                </button>
                <span class=move || format!("key-indicator {}", key_class()) title=key_label />
                <span class={status_class} />
                <span class="status-label">{status_label}</span>
            </div>
        </nav>
    }
}
