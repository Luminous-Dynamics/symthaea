// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mobile bottom navigation bar with active state highlighting.

use leptos::prelude::*;
use crate::mail_context::use_mail;

#[component]
pub fn BottomNav() -> impl IntoView {
    let mail = use_mail();
    let location = leptos_router::hooks::use_location();
    let pathname = move || location.pathname.get();

    let unread = move || {
        let count = mail.inbox.get().iter().filter(|e| !e.is_read).count();
        (count > 0).then(|| count)
    };

    let items: &[(&str, &str, &str)] = &[
        ("/", "\u{1F4E5}", "Inbox"),
        ("/compose", "\u{270F}", "Compose"),
        ("/contacts", "\u{1F465}", "Contacts"),
        ("/search", "\u{1F50D}", "Search"),
    ];

    view! {
        <nav class="bottom-nav" aria-label="Mobile navigation">
            {items.iter().map(|(href, icon, label)| {
                let href = *href;
                let icon = *icon;
                let label = *label;
                let is_inbox = href == "/";
                view! {
                    <a href=href
                       class=move || {
                           let current = pathname();
                           let active = if is_inbox {
                               current == "/" || current.is_empty()
                           } else {
                               current.starts_with(href)
                           };
                           if active { "bottom-nav-item active" } else { "bottom-nav-item" }
                       }
                       aria-current=move || {
                           let current = pathname();
                           let active = if is_inbox { current == "/" } else { current.starts_with(href) };
                           if active { "page" } else { "" }
                       }>
                        <span class="bottom-nav-icon">{icon}</span>
                        <span class="bottom-nav-label">{label}</span>
                        {is_inbox.then(|| view! {
                            {move || unread().map(|c| view! {
                                <span class="bottom-nav-badge">{c}</span>
                            })}
                        })}
                    </a>
                }
            }).collect::<Vec<_>>()}
        </nav>
    }
}
