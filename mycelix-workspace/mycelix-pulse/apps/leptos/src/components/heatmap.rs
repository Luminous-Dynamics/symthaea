// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conversation heatmap — 7x24 day/hour grid showing email activity density.

use leptos::prelude::*;
use crate::mail_context::use_mail;

#[component]
pub fn ConversationHeatmap() -> impl IntoView {
    let mail = use_mail();
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    view! {
        <div class="heatmap-section">
            <div class="heatmap-grid">
                // Header row (hours)
                <div class="heatmap-label" />
                {(0..24).map(|h| {
                    let label = if h % 6 == 0 { format!("{h}") } else { String::new() };
                    view! { <div class="heatmap-label" style="justify-content:center">{label}</div> }
                }).collect::<Vec<_>>()}
                // Data rows
                {days.iter().enumerate().map(|(day_idx, day_name)| {
                    let day_name = *day_name;
                    let cells = move || {
                        let emails = mail.inbox.get();
                        (0..24).map(|hour| {
                            let count = emails.iter().filter(|e| {
                                let d = js_sys::Date::new_0();
                                d.set_time((e.timestamp as f64) * 1000.0);
                                d.get_day() as usize == day_idx && d.get_hours() as usize == hour
                            }).count();
                            let level = match count { 0 => 0, 1 => 1, 2..=3 => 2, 4..=6 => 3, _ => 4 };
                            (hour, count, level)
                        }).collect::<Vec<_>>()
                    };
                    view! {
                        <div class="heatmap-label">{day_name}</div>
                        {move || cells().into_iter().map(|(hour, count, level)| {
                            view! {
                                <div class=format!("heatmap-cell heatmap-cell-{level}")
                                     title=format!("{day_name} {hour}:00 — {count} emails") />
                            }
                        }).collect::<Vec<_>>()}
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
