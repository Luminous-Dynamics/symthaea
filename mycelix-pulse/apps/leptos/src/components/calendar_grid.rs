// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Calendar grid views — day, week, month, mini calendar.

use leptos::prelude::*;
use mail_leptos_types::*;

/// Time grid for day/week views — shows hourly slots with positioned events.
#[component]
pub fn TimeGrid(events: Vec<CalendarEventView>, #[prop(default = 1)] columns: usize) -> impl IntoView {
    let hours: Vec<u8> = (0..24).collect();

    view! {
        <div class="time-grid" style=format!("--grid-columns: {columns}")>
            <div class="time-labels">
                {hours.iter().map(|h| {
                    let label = if *h == 0 { "12 AM".into() }
                        else if *h < 12 { format!("{h} AM") }
                        else if *h == 12 { "12 PM".into() }
                        else { format!("{} PM", h - 12) };
                    view! { <div class="time-label">{label}</div> }
                }).collect::<Vec<_>>()}
            </div>
            <div class="time-slots">
                {hours.iter().map(|h| {
                    let hour = *h;
                    view! {
                        <div class="time-slot"
                             on:dragover=|ev: web_sys::DragEvent| { ev.prevent_default(); }
                             on:drop=move |ev: web_sys::DragEvent| {
                                 ev.prevent_default();
                                 if let Some(dt) = ev.data_transfer() {
                                     let event_id = dt.get_data("text/plain").unwrap_or_default();
                                     // Store drop target in sessionStorage for the calendar page to pick up
                                     if let Some(s) = web_sys::window().and_then(|w| w.session_storage().ok().flatten()) {
                                         let _ = s.set_item("mycelix_drag_event", &format!("{}:{}", event_id, hour));
                                     }
                                     // Dispatch custom event for calendar to handle
                                     let _ = js_sys::eval(&format!(
                                         "window.dispatchEvent(new CustomEvent('mycelix-event-drop',{{detail:{{id:'{}',hour:{}}}}}))",
                                         event_id, hour
                                     ));
                                 }
                             }
                        />
                    }
                }).collect::<Vec<_>>()}

                {events.iter().filter(|e| !e.all_day).map(|evt| {
                    let title = evt.title.clone();
                    let color = evt.color.clone().unwrap_or_else(|| "#06D6C8".into());
                    let event_id = evt.id.clone();
                    let start_hour = {
                        let d = js_sys::Date::new_0();
                        d.set_time((evt.start_time as f64) * 1000.0);
                        d.get_hours() as f32 + d.get_minutes() as f32 / 60.0
                    };
                    let duration = {
                        let d = js_sys::Date::new_0();
                        d.set_time((evt.end_time as f64) * 1000.0);
                        let end = d.get_hours() as f32 + d.get_minutes() as f32 / 60.0;
                        (end - start_hour).max(0.5)
                    };
                    let top = start_hour * 48.0;
                    let height = duration * 48.0;
                    let start_str = format_grid_time(evt.start_time);
                    let end_str = format_grid_time(evt.end_time);

                    view! {
                        <div class="grid-event"
                             draggable="true"
                             on:dragstart=move |ev: web_sys::DragEvent| {
                                 if let Some(dt) = ev.data_transfer() {
                                     let _ = dt.set_data("text/plain", &event_id);
                                     dt.set_effect_allowed("move");
                                 }
                             }
                             style=format!("top: {top}px; height: {height}px; background: {color}20; border-left: 3px solid {color}; cursor: grab;")>
                            <span class="grid-event-title">{title}</span>
                            <span class="grid-event-time">{start_str}" — "{end_str}</span>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

/// Month grid — 7-column date layout with event dots.
#[component]
pub fn MonthGrid(events: Vec<CalendarEventView>, year: i32, month: u32) -> impl IntoView {
    let days_of_week = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

    let first_day = js_sys::Date::new_0();
    first_day.set_full_year(year as u32);
    first_day.set_month(month - 1);
    first_day.set_date(1);
    let start_weekday = first_day.get_day() as usize;

    let next_month = js_sys::Date::new_0();
    next_month.set_full_year(year as u32);
    next_month.set_month(month);
    next_month.set_date(0);
    let days_in_month = next_month.get_date() as usize;

    let today = js_sys::Date::new_0();
    let today_date = today.get_date() as usize;
    let today_month = today.get_month() as u32 + 1;
    let today_year = today.get_full_year() as i32;

    let rows = (start_weekday + days_in_month + 6) / 7;

    view! {
        <div class="month-grid">
            <div class="month-header-row">
                {days_of_week.iter().map(|d| view! {
                    <div class="month-header-cell">{*d}</div>
                }).collect::<Vec<_>>()}
            </div>
            {(0..rows).map(|row| view! {
                <div class="month-row">
                    {(0..7).map(|col| {
                        let idx = row * 7 + col;
                        if idx < start_weekday || idx >= start_weekday + days_in_month {
                            view! { <div class="month-cell empty" /> }.into_any()
                        } else {
                            let day = idx - start_weekday + 1;
                            let is_today = day == today_date && month == today_month && year == today_year;
                            let day_start = {
                                let d = js_sys::Date::new_0();
                                d.set_full_year(year as u32); d.set_month(month - 1); d.set_date(day as u32);
                                d.set_hours(0); d.set_minutes(0); d.set_seconds(0);
                                (d.get_time() / 1000.0) as u64
                            };
                            let day_end = day_start + 86400;
                            let day_events: Vec<_> = events.iter()
                                .filter(|e| e.start_time < day_end && e.end_time > day_start).collect();
                            let evt_count = day_events.len();

                            view! {
                                <div class=if is_today { "month-cell today" } else { "month-cell" }>
                                    <span class="month-day-num">{day}</span>
                                    {(evt_count > 0).then(|| {
                                        let dots: Vec<_> = day_events.iter().take(3).map(|e| {
                                            let c = e.color.clone().unwrap_or_else(|| "#06D6C8".into());
                                            view! { <span class="event-dot" style=format!("background:{c}") /> }
                                        }).collect();
                                        view! {
                                            <div class="month-event-dots">
                                                {dots}
                                                {(evt_count > 3).then(|| view! {
                                                    <span class="month-overflow">{format!("+{}", evt_count - 3)}</span>
                                                })}
                                            </div>
                                        }
                                    })}
                                </div>
                            }.into_any()
                        }
                    }).collect::<Vec<_>>()}
                </div>
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Mini calendar for sidebar navigation.
#[component]
pub fn MiniCalendar() -> impl IntoView {
    let now = js_sys::Date::new_0();
    let year = RwSignal::new(now.get_full_year() as i32);
    let month = RwSignal::new(now.get_month() as u32 + 1);

    let month_label = move || {
        let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        format!("{} {}", months[(month.get() - 1) as usize], year.get())
    };

    let on_prev = move |_| month.update(|m| { if *m == 1 { *m = 12; year.update(|y| *y -= 1); } else { *m -= 1; } });
    let on_next = move |_| month.update(|m| { if *m == 12 { *m = 1; year.update(|y| *y += 1); } else { *m += 1; } });

    view! {
        <div class="mini-calendar">
            <div class="mini-cal-header">
                <button class="mini-nav" on:click=on_prev>"\u{25C0}"</button>
                <span class="mini-month">{month_label}</span>
                <button class="mini-nav" on:click=on_next>"\u{25B6}"</button>
            </div>
            <MonthGrid events=vec![] year=year.get_untracked() month=month.get_untracked() />
        </div>
    }
}

fn format_grid_time(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    let h = d.get_hours();
    let m = d.get_minutes();
    if h == 0 { format!("12:{m:02} AM") }
    else if h < 12 { format!("{h}:{m:02} AM") }
    else if h == 12 { format!("12:{m:02} PM") }
    else { format!("{}:{m:02} PM", h - 12) }
}
