// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Event creation/editing form and detail popup.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use mail_leptos_types::*;
use crate::toasts::use_toasts;

#[component]
pub fn EventForm(
    #[prop(optional)] editing: Option<CalendarEventView>,
    events_signal: RwSignal<Vec<CalendarEventView>>,
    show_signal: RwSignal<bool>,
) -> impl IntoView {
    let toasts = use_toasts();
    let is_edit = editing.is_some();

    let title = RwSignal::new(editing.as_ref().map(|e| e.title.clone()).unwrap_or_default());
    let description = RwSignal::new(editing.as_ref().and_then(|e| e.description.clone()).unwrap_or_default());
    let location = RwSignal::new(editing.as_ref().and_then(|e| e.location.clone()).unwrap_or_default());
    let category = RwSignal::new(editing.as_ref().map(|e| e.category.clone()).unwrap_or_else(|| "Personal".into()));
    let all_day = RwSignal::new(editing.as_ref().map(|e| e.all_day).unwrap_or(false));
    let source = RwSignal::new(editing.as_ref().map(|e| e.source.clone()).unwrap_or(CalendarSource::Personal));

    // Date/time as strings for HTML inputs
    let now = js_sys::Date::new_0();
    let default_start = format!("{:04}-{:02}-{:02}T{:02}:{:02}",
        now.get_full_year(), now.get_month() + 1, now.get_date(),
        now.get_hours() + 1, 0);
    let default_end = format!("{:04}-{:02}-{:02}T{:02}:{:02}",
        now.get_full_year(), now.get_month() + 1, now.get_date(),
        now.get_hours() + 2, 0);

    let start_str = RwSignal::new(editing.as_ref().map(|e| ts_to_datetime_local(e.start_time)).unwrap_or(default_start));
    let end_str = RwSignal::new(editing.as_ref().map(|e| ts_to_datetime_local(e.end_time)).unwrap_or(default_end));

    let recurrence = RwSignal::new(editing.as_ref().map(|e| e.recurrence.clone()).unwrap_or(Recurrence::None));

    let on_submit = move |_| {
        let t = title.get_untracked();
        if t.trim().is_empty() {
            toasts.push("Title is required", "error");
            return;
        }

        let start_ts = datetime_local_to_ts(&start_str.get_untracked());
        let end_ts = datetime_local_to_ts(&end_str.get_untracked());

        if end_ts <= start_ts {
            toasts.push("End time must be after start time", "error");
            return;
        }

        let event = CalendarEventView {
            id: editing.as_ref().map(|e| e.id.clone()).unwrap_or_else(|| format!("cal-{}", js_sys::Date::now() as u64)),
            title: t,
            description: { let d = description.get_untracked(); if d.is_empty() { None } else { Some(d) } },
            location: { let l = location.get_untracked(); if l.is_empty() { None } else { Some(l) } },
            start_time: start_ts,
            end_time: end_ts,
            all_day: all_day.get_untracked(),
            recurrence: recurrence.get_untracked(),
            category: category.get_untracked(),
            organizer: Some("You".into()),
            attendees: vec![],
            rsvp_status: None,
            source: source.get_untracked(),
            color: None,
        };

        events_signal.update(|evts| {
            if let Some(existing) = evts.iter_mut().find(|e| e.id == event.id) {
                *existing = event;
            } else {
                evts.push(event);
            }
        });
        show_signal.set(false);
        toasts.push("Event saved", "success");
    };

    view! {
        <div class="event-form-overlay" on:click=move |_| show_signal.set(false)>
            <div class="event-form" on:click=move |ev: leptos::ev::MouseEvent| ev.stop_propagation()>
                <h2>{if is_edit { "Edit Event" } else { "New Event" }}</h2>

                <div class="form-field">
                    <label>"Title"</label>
                    <input type="text" placeholder="Event title" autofocus=true
                        prop:value=move || title.get()
                        on:input=move |ev| title.set(input_val(&ev)) />
                </div>

                <div class="form-row">
                    <div class="form-field">
                        <label>"Start"</label>
                        <input type="datetime-local"
                            prop:value=move || start_str.get()
                            on:input=move |ev| start_str.set(input_val(&ev)) />
                    </div>
                    <div class="form-field">
                        <label>"End"</label>
                        <input type="datetime-local"
                            prop:value=move || end_str.get()
                            on:input=move |ev| end_str.set(input_val(&ev)) />
                    </div>
                </div>

                <label class="option-toggle">
                    <input type="checkbox" prop:checked=move || all_day.get()
                        on:change=move |_| all_day.update(|v| *v = !*v) />
                    " All day"
                </label>

                <div class="form-field">
                    <label>"Location"</label>
                    <input type="text" placeholder="Optional location"
                        prop:value=move || location.get()
                        on:input=move |ev| location.set(input_val(&ev)) />
                </div>

                <div class="form-field">
                    <label>"Description"</label>
                    <textarea rows="3" placeholder="Optional description"
                        prop:value=move || description.get()
                        on:input=move |ev| {
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            description.set(val);
                        } />
                </div>

                <div class="form-row">
                    <div class="form-field">
                        <label>"Calendar"</label>
                        <select on:change=move |ev| {
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            source.set(match val.as_str() {
                                "Community" => CalendarSource::Community,
                                "Hearth" => CalendarSource::Hearth,
                                _ => CalendarSource::Personal,
                            });
                        }>
                            <option value="Personal" selected=true>"Personal"</option>
                            <option value="Community">"Community"</option>
                            <option value="Hearth">"Hearth"</option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label>"Recurrence"</label>
                        <select on:change=move |ev| {
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            recurrence.set(match val.as_str() {
                                "Daily" => Recurrence::Daily,
                                "Weekly" => Recurrence::Weekly,
                                "Monthly" => Recurrence::Monthly,
                                "Yearly" => Recurrence::Yearly,
                                _ => Recurrence::None,
                            });
                        }>
                            <option value="None">"No repeat"</option>
                            <option value="Daily">"Daily"</option>
                            <option value="Weekly">"Weekly"</option>
                            <option value="Monthly">"Monthly"</option>
                            <option value="Yearly">"Yearly"</option>
                        </select>
                    </div>
                </div>

                <div class="form-actions">
                    <button class="btn btn-secondary" on:click=move |_| show_signal.set(false)>"Cancel"</button>
                    <button class="btn btn-primary" on:click=on_submit>
                        {if is_edit { "Save Changes" } else { "Create Event" }}
                    </button>
                </div>
            </div>
        </div>
    }
}

fn input_val(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

fn ts_to_datetime_local(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}",
        d.get_full_year(), d.get_month() + 1, d.get_date(),
        d.get_hours(), d.get_minutes())
}

fn datetime_local_to_ts(s: &str) -> u64 {
    // Parse "YYYY-MM-DDTHH:MM" format
    let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_str(s));
    (d.get_time() / 1000.0) as u64
}
