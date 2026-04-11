// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Email card with checkbox (#3 batch), drag support (#7), trust ring (#6), inline actions (#2).

use leptos::prelude::*;
use mail_leptos_types::*;

use crate::components::{EncryptionBadge, TrustGateBadge};
use crate::mail_context::use_mail;
use crate::keyboard::use_keyboard;

#[component]
pub fn EmailCard(email: EmailListItem, #[prop(optional)] index: Option<usize>) -> impl IntoView {
    let mail = use_mail();
    let kb = use_keyboard();
    let hash = email.hash.clone();
    let sender = email.sender.clone();
    let sender_name = email.sender_name.clone().unwrap_or_else(|| {
        let s = &email.sender;
        if s.len() > 12 { format!("{}...{}", &s[..6], &s[s.len()-6..]) }
        else { s.clone() }
    });
    let subject = email.subject.clone().unwrap_or_else(|| "(encrypted)".to_string());
    let snippet = email.snippet.clone().unwrap_or_default();
    let is_read = email.is_read;
    let is_starred = email.is_starred;
    let star_icon = email.star_type.map(|s| s.icon()).unwrap_or("\u{2606}");
    let email_labels = email.labels.clone();
    let has_attachments = email.has_attachments;
    let priority = email.priority.clone();
    let crypto = email.crypto_suite.clone();
    let thread_count = email.thread_id.as_ref().map(|tid| {
        mail.inbox.get_untracked().iter().filter(|e| e.thread_id.as_deref() == Some(tid)).count()
    });

    // Consistent avatar color per sender (deterministic hash)
    let avatar_color = {
        let colors = ["#06D6C8", "#8b7ec8", "#f59e0b", "#4ade80", "#ec4899", "#3b82f6", "#ef4444", "#f97316"];
        let hash_val: usize = sender.bytes().fold(0usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize));
        colors[hash_val % colors.len()]
    };

    let trust = mail.trust_for_sender(&sender);
    let trust_class = trust.map(|s| {
        if s >= 0.8 { "trust-ring-high" }
        else if s >= 0.5 { "trust-ring-medium" }
        else { "trust-ring-low" }
    }).unwrap_or("trust-ring-none");

    let time_display = {
        let ts = email.timestamp;
        let now = js_sys::Date::now() as u64 / 1000;
        let diff = now.saturating_sub(ts);
        if diff < 3600 { format!("{}m", diff / 60) }
        else if diff < 86400 { format!("{}h", diff / 3600) }
        else { format!("{}d", diff / 86400) }
    };

    let read_class = if is_read { "email-card read" } else { "email-card unread" };
    let priority_class = priority.css_class();
    let navigate_to = format!("/read/{hash}");
    let aria = format!("Email from {sender_name}: {subject}");

    // Keyboard focus (#1)
    let is_kb_focused = move || {
        index.and_then(|i| kb.focused_index.get().filter(|&fi| fi == i)).is_some()
    };

    // Batch selection (#3)
    let hash_sel = hash.clone();
    let mail_sel = mail.clone();
    let hash_sel2 = hash.clone();
    let mail_sel2 = mail.clone();
    let is_selected = move || mail_sel.selected_hashes.get().contains(&hash_sel);
    let is_selected2 = move || mail_sel2.selected_hashes.get().contains(&hash_sel2);
    let hash_toggle = hash.clone();
    let mail_toggle = mail.clone();
    let on_checkbox = move |ev: leptos::ev::MouseEvent| {
        ev.prevent_default();
        ev.stop_propagation();
        mail_toggle.toggle_selection(&hash_toggle);
    };

    // Inline actions — cycle star (#11)
    let hash_star = hash.clone();
    let mail_star = mail.clone();
    let mail_read = mail.clone();
    let hash_read = hash.clone();
    let on_star = move |ev: leptos::ev::MouseEvent| {
        ev.prevent_default();
        ev.stop_propagation();
        mail_star.cycle_star(&hash_star);
    };
    let on_toggle_read = move |ev: leptos::ev::MouseEvent| {
        ev.prevent_default();
        ev.stop_propagation();
        mail_read.toggle_read(&hash_read);
    };

    // Drag support (#7)
    let hash_drag = hash.clone();
    let on_dragstart = move |ev: web_sys::DragEvent| {
        if let Some(dt) = ev.data_transfer() {
            let _ = dt.set_data("text/plain", &hash_drag);
            dt.set_effect_allowed("move");
        }
    };

    // Label drop target
    let hash_label_drop = hash.clone();
    let mail_label_drop = mail.clone();
    let on_label_dragover = move |ev: web_sys::DragEvent| {
        if let Some(dt) = ev.data_transfer() {
            if dt.types().includes(&"application/x-label".into(), 0) {
                ev.prevent_default();
            }
        }
    };
    let on_label_drop = move |ev: web_sys::DragEvent| {
        ev.prevent_default();
        if let Some(dt) = ev.data_transfer() {
            if let Ok(label) = dt.get_data("application/x-label") {
                if !label.is_empty() {
                    mail_label_drop.apply_label(&hash_label_drop, &label);
                }
            }
        }
    };

    // Mobile swipe gestures (#20)
    let swipe_start_x = RwSignal::new(0.0f64);
    let swipe_offset = RwSignal::new(0.0f64);
    let swiping = RwSignal::new(false);
    let hash_swipe = hash.clone();
    let mail_swipe_archive = mail.clone();
    let mail_swipe_delete = mail.clone();
    let hash_swipe_archive = hash_swipe.clone();
    let hash_swipe_delete = hash_swipe.clone();

    let on_touchstart = move |ev: web_sys::TouchEvent| {
        if let Some(touch) = ev.touches().get(0) {
            swipe_start_x.set(touch.client_x() as f64);
            swipe_offset.set(0.0);
            swiping.set(true);
        }
    };
    let on_touchmove = move |ev: web_sys::TouchEvent| {
        if !swiping.get() { return; }
        if let Some(touch) = ev.touches().get(0) {
            let dx = touch.client_x() as f64 - swipe_start_x.get();
            if dx.abs() > 10.0 { ev.prevent_default(); }
            swipe_offset.set(dx);
        }
    };
    let on_touchend = move |_: web_sys::TouchEvent| {
        if !swiping.get() { return; }
        swiping.set(false);
        let offset = swipe_offset.get();
        if offset > 80.0 {
            mail_swipe_archive.archive_email(&hash_swipe_archive);
        } else if offset < -80.0 {
            mail_swipe_delete.delete_email(&hash_swipe_delete);
        }
        swipe_offset.set(0.0);
    };

    view! {
        <a href=navigate_to
           class=move || {
               let mut cls = format!("{read_class} {priority_class}");
               if is_kb_focused() { cls.push_str(" kb-focused"); }
               if is_selected() { cls.push_str(" selected"); }
               cls
           }
           aria-label=aria
           draggable="true"
           on:dragstart=on_dragstart
           on:dragover=on_label_dragover
           on:drop=on_label_drop
           on:touchstart=on_touchstart
           on:touchmove=on_touchmove
           on:touchend=on_touchend
           style=move || {
               let offset = swipe_offset.get();
               if offset.abs() > 5.0 {
                   format!("transform:translateX({}px);transition:none;", offset)
               } else {
                   "transition:transform 0.2s ease;".to_string()
               }
           }>
            <div class="email-card-left">
                <input
                    type="checkbox"
                    class="email-checkbox"
                    prop:checked=is_selected2
                    on:click=on_checkbox
                />
                <div class=format!("sender-avatar {trust_class}")
                     style=format!("color: {avatar_color}")
                     title=move || trust.map(|s| format!("Trust: {:.0}%", s * 100.0)).unwrap_or_default()>
                    {sender_name.chars().next().unwrap_or('?').to_uppercase().to_string()}
                </div>
            </div>
            <div class="email-card-body">
                <div class="email-card-header">
                    <span class="sender-name">{sender_name.clone()}
                        // Verified badge — shown when sender has confirmed DID in identity cluster
                        {trust.filter(|s| *s >= 0.8).map(|_| view! {
                            <span class="verified-badge" title="Verified identity">{"\u{2705}"}</span>
                        })}
                    </span>
                    <span class="email-time">{time_display}</span>
                </div>
                <div class="email-subject">
                    {subject.clone()}
                    {thread_count.and_then(|c| (c > 1).then(|| view! {
                        <span class="thread-count">{format!(" ({c})")}</span>
                    }))}
                </div>
                {(!email_labels.is_empty()).then(|| {
                    let labels = email_labels.clone();
                    view! {
                        <div class="email-labels">
                            {labels.into_iter().map(|l| view! {
                                <span class="email-label-chip">{l}</span>
                            }).collect::<Vec<_>>()}
                        </div>
                    }
                })}
                <div class="email-snippet">{snippet.clone()}</div>
            </div>
            <div class="email-card-right">
                <div class="email-indicators">
                    <button class=move || if is_starred { "star-btn starred" } else { "star-btn" }
                            on:click=on_star title="Cycle star type (click multiple times)">
                        {star_icon}
                    </button>
                    <button class="read-btn" on:click=on_toggle_read
                            title=if is_read { "Mark unread" } else { "Mark read" }>
                        {if is_read { "\u{2709}" } else { "\u{1F4E9}" }}
                    </button>
                    {has_attachments.then(|| view! { <span class="attachment-indicator" title="Has attachments">"\u{1F4CE}"</span> })}
                    {(priority != EmailPriority::Normal).then(|| view! {
                        <span class=format!("priority-indicator {}", priority.css_class()) title=priority.label()>
                            {match priority {
                                EmailPriority::Urgent => "!",
                                EmailPriority::High => "\u{2191}",
                                EmailPriority::Low => "\u{2193}",
                                _ => "",
                            }}
                        </span>
                    })}
                </div>
                // Expiry countdown for revocable emails
                {
                    // Demo: emails with "revoke" label or snoozed have expiry
                    let has_expiry = email.is_snoozed || email.labels.iter().any(|l| l.to_lowercase().contains("expir"));
                    has_expiry.then(|| {
                        let ts = email.timestamp;
                        let expiry_ts = ts + 86400 * 7; // 7 days from send
                        let now = js_sys::Date::now() as u64 / 1000;
                        let remaining = expiry_ts.saturating_sub(now);
                        let hours_left = remaining / 3600;
                        let days_left = hours_left / 24;
                        let expiry_class = if days_left < 1 { "expiry-urgent" } else if days_left < 3 { "expiry-warning" } else { "expiry-normal" };
                        let label = if days_left > 0 { format!("{days_left}d") } else { format!("{hours_left}h") };
                        view! {
                            <span class=format!("expiry-badge {expiry_class}") title=format!("Key expires in {days_left} days {hours_left}h")>
                                "\u{23F3} "{label}
                            </span>
                        }
                    })
                }
                <crate::components::AssuranceBadge />
                <crate::components::KarmaBadge sender=sender.clone() />
                <TrustGateBadge sender=sender.clone() trust_score=trust />
                <EncryptionBadge crypto=crypto />
            </div>
        </a>
    }
}
