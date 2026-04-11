// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Notification center — bell icon dropdown with unified activity feed.

use leptos::prelude::*;
use crate::mail_context::use_mail;

#[derive(Clone, Debug)]
pub struct NotifItem {
    pub icon: &'static str,
    pub title: String,
    pub detail: String,
    pub category: &'static str,
    pub timestamp: u64,
}

#[component]
pub fn NotificationCenter() -> impl IntoView {
    let mail = use_mail();
    let open = RwSignal::new(false);
    let filter = RwSignal::new("all"); // "all", "mail", "chat", "calendar", "trust"

    let notifications = move || {
        let mut items = Vec::<NotifItem>::new();
        let now = js_sys::Date::now() as u64 / 1000;

        // Unread emails
        for e in mail.inbox.get().iter().filter(|e| !e.is_read).take(5) {
            let sender = e.sender_name.clone().unwrap_or_else(|| e.sender[..8.min(e.sender.len())].to_string());
            items.push(NotifItem {
                icon: "\u{2709}",
                title: format!("New mail from {sender}"),
                detail: e.subject.clone().unwrap_or_else(|| "(encrypted)".into()),
                category: "mail",
                timestamp: e.timestamp,
            });
        }

        // Trust gate actions (staked/quarantined emails)
        for e in mail.inbox.get().iter().take(20) {
            if let Some(score) = mail.sender_trust.get().get(&e.sender) {
                if *score < 0.5 {
                    items.push(NotifItem {
                        icon: "\u{1F6E1}",
                        title: "Trust review needed".into(),
                        detail: format!("Email from {} (trust: {:.0}%)",
                            e.sender_name.clone().unwrap_or_else(|| "unknown".into()),
                            score * 100.0),
                        category: "trust",
                        timestamp: e.timestamp,
                    });
                }
            }
        }

        // Calendar reminders (events starting within 1 hour)
        // These would come from the calendar context in production
        items.push(NotifItem {
            icon: "\u{1F4C5}",
            title: "Upcoming event".into(),
            detail: "Team standup in 30 minutes".into(),
            category: "calendar",
            timestamp: now,
        });

        // Sort by timestamp descending
        items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Apply filter
        let f = filter.get();
        if f != "all" {
            items.retain(|i| i.category == f);
        }
        items
    };

    let total_count = move || {
        let unread = mail.inbox.get().iter().filter(|e| !e.is_read).count();
        let trust_pending = mail.inbox.get().iter().filter(|e| {
            mail.sender_trust.get().get(&e.sender).map(|s| *s < 0.5).unwrap_or(false)
        }).count();
        unread + trust_pending
    };

    view! {
        <div class="notif-center-wrapper">
            <button class="notif-bell" on:click=move |_| open.update(|v| *v = !*v)
                    title="Notifications">
                "\u{1F514}"
                {move || {
                    let count = total_count();
                    (count > 0).then(|| view! {
                        <span class="notif-badge">{count}</span>
                    })
                }}
            </button>
            <div class="notif-dropdown" style=move || if open.get() { "" } else { "display:none" }>
                <div class="notif-header">
                    <h3>"Notifications"</h3>
                    <button class="notif-close" on:click=move |_| open.set(false)>"\u{2715}"</button>
                </div>
                <div class="notif-filters">
                    {["all", "mail", "chat", "calendar", "trust"].iter().map(|f| {
                        let f = *f;
                        let label = match f { "all" => "All", "mail" => "\u{2709}", "chat" => "\u{1F4AC}", "calendar" => "\u{1F4C5}", "trust" => "\u{1F6E1}", _ => f };
                        view! {
                            <button class=move || if filter.get() == f { "notif-filter active" } else { "notif-filter" }
                                    on:click=move |_| filter.set(f)>{label}</button>
                        }
                    }).collect::<Vec<_>>()}
                </div>
                <div class="notif-list">
                    {move || {
                        let items = notifications();
                        if items.is_empty() {
                            view! { <p class="notif-empty">"No notifications"</p> }.into_any()
                        } else {
                            view! {
                                <div>
                                    {items.into_iter().map(|item| {
                                        view! {
                                            <div class=format!("notif-item notif-{}", item.category)>
                                                <span class="notif-icon">{item.icon}</span>
                                                <div class="notif-content">
                                                    <span class="notif-title">{item.title}</span>
                                                    <span class="notif-detail">{item.detail}</span>
                                                </div>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
            </div>
        </div>
    }
}
