// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;

#[component]
pub fn CalendarPage() -> impl IntoView {
    let commons = use_commons();
    view! {
        <div class="calendar-page" data-page="calendar" role="main">
            <h1 class="page-title">"Community Calendar"</h1>
            <p class="page-subtitle">"gatherings, work days, celebrations"</p>
            <div class="events-list" data-section="events" role="list">
                {move || {
                    let events = commons.events.get();
                    if events.is_empty() {
                        view! { <p class="empty-state">"no upcoming events"</p> }.into_any()
                    } else {
                        events.into_iter().map(|e| {
                            let organizer = e.organizer_did.split(':').last().unwrap_or(&e.organizer_did).to_string();
                            let cat_display = e.category.clone();
                            let date = {
                                let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(e.start_time as f64 / 1000.0));
                                format!("{}/{}/{}", d.get_date(), d.get_month() + 1, d.get_full_year())
                            };
                            view! {
                                <div class="event-card" data-event-hash=e.hash.clone() data-category=e.category.clone()
                                    data-start-time=e.start_time.to_string() data-rsvp-count=e.rsvp_count.to_string() role="listitem">
                                    <div class="event-header">
                                        <h3 class="event-title">{e.title}</h3>
                                        <span class="event-date">{date}</span>
                                    </div>
                                    <p class="event-desc">{e.description}</p>
                                    <div class="event-meta">
                                        <span class="event-category">{cat_display}</span>
                                        <span data-metric="rsvps">{format!("{}/{} attending", e.rsvp_count, e.max_attendees)}</span>
                                        <span>{format!("by {organizer}")}</span>
                                    </div>
                                </div>
                            }
                        }).collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}
