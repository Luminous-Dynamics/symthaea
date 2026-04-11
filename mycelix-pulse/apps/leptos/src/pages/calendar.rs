// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Calendar page — personal + community + hearth events.
//! Day/week/month + agenda views. Event CRUD. Three-source unified calendar.

use leptos::prelude::*;
use mail_leptos_types::*;
use crate::toasts::use_toasts;
use crate::components::{TimeGrid, MonthGrid, EventForm};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CalView { Day, Week, Month, Agenda }

#[component]
pub fn CalendarPage() -> impl IntoView {
    let toasts = use_toasts();
    let events = RwSignal::new(crate::mock_data::mock_calendar_events());
    let view_mode = RwSignal::new(CalView::Agenda);
    let show_personal = RwSignal::new(true);
    let show_community = RwSignal::new(true);
    let show_hearth = RwSignal::new(true);
    let show_form = RwSignal::new(false);
    let editing_event = RwSignal::new(Option::<String>::None);
    let edit_title = RwSignal::new(String::new());
    let edit_desc = RwSignal::new(String::new());
    let edit_location = RwSignal::new(String::new());
    let toasts_edit_overlay = toasts.clone();

    let now = js_sys::Date::new_0();
    let current_year = RwSignal::new(now.get_full_year() as i32);
    let current_month = RwSignal::new(now.get_month() as u32 + 1);

    let filtered_events = move || {
        events.get().into_iter().filter(|e| {
            match &e.source {
                CalendarSource::Personal => show_personal.get(),
                CalendarSource::Community => show_community.get(),
                CalendarSource::Hearth => show_hearth.get(),
                CalendarSource::External(_) => true,
            }
        }).collect::<Vec<_>>()
    };

    let upcoming = move || {
        let mut evts = filtered_events();
        evts.sort_by_key(|e| e.start_time);
        evts
    };

    // Event form uses signals directly — no closure callbacks needed

    let month_label = move || {
        let months = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"];
        format!("{} {}", months[(current_month.get() - 1) as usize], current_year.get())
    };

    let on_prev = move |_| current_month.update(|m| { if *m == 1 { *m = 12; current_year.update(|y| *y -= 1); } else { *m -= 1; } });
    let on_next = move |_| current_month.update(|m| { if *m == 12 { *m = 1; current_year.update(|y| *y += 1); } else { *m += 1; } });
    let on_today = move |_| {
        let now = js_sys::Date::new_0();
        current_year.set(now.get_full_year() as i32);
        current_month.set(now.get_month() as u32 + 1);
    };

    // Natural language event input
    let nlp_input = RwSignal::new(String::new());
    let toasts_nlp = toasts.clone();
    let on_nlp_create = move |_| {
        let text = nlp_input.get_untracked();
        if text.trim().is_empty() { return; }
        if let Some(parsed) = crate::nlp_time::parse_natural_event(&text) {
            let now_ts = (js_sys::Date::now() / 1000.0) as u64;
            let start = now_ts + (parsed.start_offset_hours * 3600.0) as u64;
            let end = start + (parsed.duration_hours * 3600.0) as u64;

            // Conflict detection — check for overlapping events
            let conflicts: Vec<String> = events.get_untracked().iter()
                .filter(|e| e.start_time < end && e.end_time > start && !e.all_day)
                .map(|e| e.title.clone())
                .collect();
            if !conflicts.is_empty() {
                let conflict_names = conflicts.join(", ");
                let confirmed = web_sys::window()
                    .and_then(|w| w.confirm_with_message(
                        &format!("This event overlaps with: {conflict_names}. Create anyway?")
                    ).ok())
                    .unwrap_or(true);
                if !confirmed { return; }
            }

            events.update(|evts| {
                evts.push(CalendarEventView {
                    id: format!("nlp-{}", js_sys::Date::now() as u64),
                    title: parsed.title,
                    description: None,
                    location: None,
                    start_time: start, end_time: end,
                    all_day: parsed.duration_hours >= 24.0,
                    recurrence: parsed.recurrence,
                    category: "Personal".into(),
                    organizer: Some("You".into()),
                    attendees: vec![],
                    rsvp_status: None,
                    source: CalendarSource::Personal,
                    color: Some("#06D6C8".into()),
                });
            });
            nlp_input.set(String::new());
            toasts_nlp.push("Event created from natural language", "success");
        } else {
            toasts_nlp.push("Couldn't parse that — try 'Meeting tomorrow at 2pm'", "error");
        }
    };

    view! {
        <div class="page page-calendar">
            <div class="page-header">
                <div class="cal-title-row">
                    <h1>{month_label}</h1>
                    <div class="cal-nav">
                        <button class="btn btn-sm btn-secondary" on:click=on_today>"Today"</button>
                        <button class="btn btn-sm btn-secondary" on:click=on_prev>"\u{25C0}"</button>
                        <button class="btn btn-sm btn-secondary" on:click=on_next>"\u{25B6}"</button>
                    </div>
                </div>
                <div class="cal-controls">
                    <div class="view-toggles">
                        <button class=move || if view_mode.get() == CalView::Agenda { "view-btn active" } else { "view-btn" }
                            on:click=move |_| view_mode.set(CalView::Agenda)>"Agenda"</button>
                        <button class=move || if view_mode.get() == CalView::Day { "view-btn active" } else { "view-btn" }
                            on:click=move |_| view_mode.set(CalView::Day)>"Day"</button>
                        <button class=move || if view_mode.get() == CalView::Week { "view-btn active" } else { "view-btn" }
                            on:click=move |_| view_mode.set(CalView::Week)>"Week"</button>
                        <button class=move || if view_mode.get() == CalView::Month { "view-btn active" } else { "view-btn" }
                            on:click=move |_| view_mode.set(CalView::Month)>"Month"</button>
                    </div>
                    <button class="btn btn-primary btn-sm" on:click=move |_| show_form.set(true)>
                        "+ New Event"
                    </button>
                </div>
            </div>

            // Natural language event creation
            <div class="nlp-event-bar">
                <input
                    class="nlp-input"
                    type="text"
                    placeholder="Quick add: 'Meeting tomorrow at 2pm' or 'Lunch with Alice Friday'"
                    prop:value=move || nlp_input.get()
                    on:input=move |ev| {
                        use wasm_bindgen::JsCast;
                        let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                        nlp_input.set(val);
                    }
                    on:keydown={
                        let toasts_nlp2 = toasts.clone();
                        let events2 = events.clone();
                        move |ev: web_sys::KeyboardEvent| {
                            if ev.key() == "Enter" {
                                let text = nlp_input.get_untracked();
                                if let Some(parsed) = crate::nlp_time::parse_natural_event(&text) {
                                    let now_ts = (js_sys::Date::now() / 1000.0) as u64;
                                    let start = now_ts + (parsed.start_offset_hours * 3600.0) as u64;
                                    let end = start + (parsed.duration_hours * 3600.0) as u64;
                                    events2.update(|evts| {
                                        evts.push(CalendarEventView {
                                            id: format!("nlp-{}", js_sys::Date::now() as u64),
                                            title: parsed.title, description: None, location: None,
                                            start_time: start, end_time: end,
                                            all_day: parsed.duration_hours >= 24.0,
                                            recurrence: parsed.recurrence,
                                            category: "Personal".into(), organizer: Some("You".into()),
                                            attendees: vec![], rsvp_status: None,
                                            source: CalendarSource::Personal, color: Some("#06D6C8".into()),
                                        });
                                    });
                                    nlp_input.set(String::new());
                                    toasts_nlp2.push("Event created", "success");
                                }
                            }
                        }
                    }
                />
                <button class="btn btn-primary btn-sm" on:click=move |_| on_nlp_create(())>"Add"</button>
            </div>

            // Source filters
            <div class="cal-filters">
                <label class="cal-source-toggle">
                    <input type="checkbox" prop:checked=move || show_personal.get()
                        on:change=move |_| show_personal.update(|v| *v = !*v) />
                    <span class="source-dot" style="background: #06D6C8" />
                    " My Time"
                </label>
                <label class="cal-source-toggle">
                    <input type="checkbox" prop:checked=move || show_community.get()
                        on:change=move |_| show_community.update(|v| *v = !*v) />
                    <span class="source-dot" style="background: #8b7ec8" />
                    " Commons"
                </label>
                <label class="cal-source-toggle">
                    <input type="checkbox" prop:checked=move || show_hearth.get()
                        on:change=move |_| show_hearth.update(|v| *v = !*v) />
                    <span class="source-dot" style="background: #f59e0b" />
                    " Hearth"
                </label>
            </div>

            // Calendar views
            {move || match view_mode.get() {
                CalView::Month => {
                    let evts = filtered_events();
                    let y = current_year.get();
                    let m = current_month.get();
                    view! { <MonthGrid events=evts year=y month=m /> }.into_any()
                }
                CalView::Day | CalView::Week => {
                    let evts = filtered_events();
                    let cols = if view_mode.get() == CalView::Week { 7 } else { 1 };
                    view! { <TimeGrid events=evts columns=cols /> }.into_any()
                }
                CalView::Agenda => {
                    let evts = upcoming();
                    if evts.is_empty() {
                        view! {
                            <div class="empty-state">
                                <span class="empty-icon">"\u{1F4C5}"</span>
                                <p class="empty-title">"No upcoming events"</p>
                                <p class="empty-desc">"Create an event to get started."</p>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div class="event-list">
                                {evts.into_iter().map(|evt| {
                                    let color = evt.color.clone().unwrap_or_else(|| "#06D6C8".into());
                                    let title = evt.title.clone();
                                    let source_label = evt.source.label();
                                    let location = evt.location.clone();
                                    let organizer = evt.organizer.clone().unwrap_or_default();
                                    let attendee_count = evt.attendees.len();
                                    let all_day = evt.all_day;
                                    let rsvp = evt.rsvp_status.clone();
                                    let start = format_time(evt.start_time);
                                    let end = format_time(evt.end_time);
                                    let date = format_date(evt.start_time);
                                    let desc = evt.description.clone().unwrap_or_default();
                                    let recurrence = evt.recurrence.clone();

                                    let id_edit = evt.id.clone();
                                    let title_edit = evt.title.clone();
                                    let desc_edit = evt.description.clone().unwrap_or_default();
                                    let loc_edit = evt.location.clone().unwrap_or_default();
                                    let id_del = evt.id.clone();
                                    let id_g = evt.id.clone(); let id_m = evt.id.clone(); let id_n = evt.id.clone();
                                    let t1 = toasts.clone(); let t2 = toasts.clone(); let t3 = toasts.clone();
                                    let t_del = toasts.clone();

                                    view! {
                                        <div class="cal-event-card" style=format!("border-left: 4px solid {color}")>
                                            <div class="cal-event-header">
                                                <div class="cal-event-time">
                                                    <span class="cal-date">{date}</span>
                                                    {if all_day {
                                                        view! { <span class="cal-time">"All day"</span> }.into_any()
                                                    } else {
                                                        view! { <span class="cal-time">{start}" — "{end}</span> }.into_any()
                                                    }}
                                                </div>
                                                <span class="cal-source">{source_label}</span>
                                            </div>
                                            <h3 class="cal-event-title">{title}</h3>
                                            {(!desc.is_empty()).then(|| view! { <p class="cal-event-desc">{desc.clone()}</p> })}
                                            {location.map(|loc| view! { <p class="cal-event-location">"\u{1F4CD} "{loc}</p> })}
                                            <div class="cal-event-meta">
                                                {(!organizer.is_empty()).then(|| view! { <span>"By "{organizer.clone()}</span> })}
                                                {(attendee_count > 0).then(|| view! { <span>{format!("{attendee_count} attendee(s)")}</span> })}
                                                {(recurrence != Recurrence::None).then(|| view! { <span>{format!("{:?}", recurrence)}</span> })}
                                            </div>
                                            <div class="cal-event-actions">
                                                <button class="btn btn-sm btn-secondary" on:click=move |_| {
                                                    edit_title.set(title_edit.clone());
                                                    edit_desc.set(desc_edit.clone());
                                                    edit_location.set(loc_edit.clone());
                                                    editing_event.set(Some(id_edit.clone()));
                                                }>"\u{270F} Edit"</button>
                                                <button class="btn btn-sm btn-secondary" on:click=move |_| {
                                                    events.update(|evts| evts.retain(|e| e.id != id_del));
                                                    t_del.push("Event deleted", "info");
                                                }>"\u{1F5D1} Delete"</button>
                                            </div>
                                            {rsvp.map(|status| view! {
                                                <div class="cal-rsvp">
                                                    <span class=format!("rsvp-current {}", status.css_class())>{status.label()}</span>
                                                    <div class="rsvp-buttons">
                                                        <button class="btn btn-sm rsvp-going" on:click=move |_| {
                                                            events.update(|evts| { if let Some(e) = evts.iter_mut().find(|e| e.id == id_g) { e.rsvp_status = Some(RsvpStatus::Going); } });
                                                            t1.push("RSVP: Going", "success");
                                                        }>"Going"</button>
                                                        <button class="btn btn-sm rsvp-maybe" on:click=move |_| {
                                                            events.update(|evts| { if let Some(e) = evts.iter_mut().find(|e| e.id == id_m) { e.rsvp_status = Some(RsvpStatus::Maybe); } });
                                                            t2.push("RSVP: Maybe", "info");
                                                        }>"Maybe"</button>
                                                        <button class="btn btn-sm rsvp-no" on:click=move |_| {
                                                            events.update(|evts| { if let Some(e) = evts.iter_mut().find(|e| e.id == id_n) { e.rsvp_status = Some(RsvpStatus::NotGoing); } });
                                                            t3.push("RSVP: No", "info");
                                                        }>"No"</button>
                                                    </div>
                                                </div>
                                            })}
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        }.into_any()
                    }
                }
            }}

            // Edit event overlay
            {move || editing_event.get().map(|eid| {
                let eid_save = eid.clone();
                let toasts_edit = toasts_edit_overlay.clone();
                view! {
                    <div class="event-edit-overlay">
                        <div class="event-edit-card">
                            <h3>"Edit Event"</h3>
                            <div class="form-field">
                                <label>"Title"</label>
                                <input type="text" prop:value=move || edit_title.get()
                                    on:input=move |ev| {
                                        use wasm_bindgen::JsCast;
                                        let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                        edit_title.set(val);
                                    } />
                            </div>
                            <div class="form-field">
                                <label>"Description"</label>
                                <textarea prop:value=move || edit_desc.get()
                                    on:input=move |ev| {
                                        use wasm_bindgen::JsCast;
                                        let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                        edit_desc.set(val);
                                    } />
                            </div>
                            <div class="form-field">
                                <label>"Location"</label>
                                <input type="text" prop:value=move || edit_location.get()
                                    on:input=move |ev| {
                                        use wasm_bindgen::JsCast;
                                        let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                        edit_location.set(val);
                                    } />
                            </div>
                            <div class="event-edit-buttons">
                                <button class="btn btn-primary" on:click=move |_| {
                                    let new_title = edit_title.get_untracked();
                                    let new_desc = edit_desc.get_untracked();
                                    let new_loc = edit_location.get_untracked();
                                    events.update(|evts| {
                                        if let Some(e) = evts.iter_mut().find(|e| e.id == eid_save) {
                                            e.title = new_title;
                                            e.description = if new_desc.is_empty() { None } else { Some(new_desc) };
                                            e.location = if new_loc.is_empty() { None } else { Some(new_loc) };
                                        }
                                    });
                                    editing_event.set(None);
                                    toasts_edit.push("Event updated", "success");
                                }>"Save"</button>
                                <button class="btn btn-secondary" on:click=move |_| editing_event.set(None)>"Cancel"</button>
                            </div>
                        </div>
                    </div>
                }
            })}

            // Event form overlay
            {move || show_form.get().then(|| view! {
                <EventForm events_signal=events show_signal=show_form />
            })}
        </div>
    }
}

fn format_time(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    format!("{:02}:{:02}", d.get_hours(), d.get_minutes())
}

fn format_date(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{} {} {}", days[d.get_day() as usize], months[d.get_month() as usize], d.get_date())
}
