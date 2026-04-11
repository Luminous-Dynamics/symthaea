// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dual sidebar — The Record (async mail) + The Pulse (real-time chat) + Spaces.

use leptos::prelude::*;
use crate::mail_context::use_mail;
use crate::toasts::use_toasts;

#[component]
pub fn Sidebar() -> impl IntoView {
    let mail = use_mail();
    let toasts = use_toasts();
    let chat_channels = RwSignal::new(crate::mock_data::mock_chat_channels());

    let compose = move |_| {
        mail.compose_mode.set(mail_leptos_types::ComposeMode::New);
        let navigate = leptos_router::hooks::use_navigate();
        navigate("/compose", Default::default());
    };

    let drag_over_folder = RwSignal::new(Option::<String>::None);
    let mail_trust_sidebar = mail.clone();
    let mail_att_sidebar = mail.clone();

    view! {
        <aside class="sidebar" aria-label="Navigation" role="navigation">
            <button class="compose-btn" on:click=compose>
                <span class="compose-icon">"+"</span>
                " Compose"
            </button>

            // ── THE RECORD (async, DHT-permanent) ──
            <div class="sidebar-section">
                <h4 class="sidebar-section-title">"The Record"</h4>
                <div class="folder-list">
                    {move || mail.folders.get().iter().map(|f| {
                        let name = f.name.clone();
                        let folder_hash = f.hash.clone();
                        let icon = f.icon();
                        let unread = f.unread_count;
                        let name_active = name.clone();
                        let is_active = move || mail.active_folder.get() == name_active;
                        let name_set = name.clone();
                        let set_active = move |_| {
                            mail.active_folder.set(name_set.clone());
                            let navigate = leptos_router::hooks::use_navigate();
                            navigate("/", Default::default());
                        };

                        let name_display = name.clone();
                        let is_drag_over = {
                            let fh = folder_hash.clone();
                            move || drag_over_folder.get().as_deref() == Some(&fh)
                        };

                        let fh_over = folder_hash.clone();
                        let on_dragover = move |ev: web_sys::DragEvent| {
                            ev.prevent_default();
                            drag_over_folder.set(Some(fh_over.clone()));
                        };
                        let on_dragleave = move |_: web_sys::DragEvent| {
                            drag_over_folder.set(None);
                        };
                        let fh_drop = folder_hash.clone();
                        let mail_drop = mail.clone();
                        let toasts_drop = toasts.clone();
                        let name_drop = name.clone();
                        let on_drop = move |ev: web_sys::DragEvent| {
                            ev.prevent_default();
                            drag_over_folder.set(None);
                            if let Some(dt) = ev.data_transfer() {
                                if let Ok(hash) = dt.get_data("text/plain") {
                                    if !hash.is_empty() {
                                        mail_drop.move_to_folder(&hash, &fh_drop);
                                        toasts_drop.push(format!("Moved to {}", name_drop), "info");
                                    }
                                }
                            }
                        };

                        view! {
                            <button
                                class=move || {
                                    if is_drag_over() { "folder-item drag-over" }
                                    else if is_active() { "folder-item active" }
                                    else { "folder-item" }
                                }
                                on:click=set_active
                                on:dragover=on_dragover
                                on:dragleave=on_dragleave
                                on:drop=on_drop
                            >
                                <span class="folder-icon">{icon}</span>
                                <span class="folder-name">{name_display}</span>
                                {(unread > 0).then(|| view! {
                                    <span class="folder-badge">{unread}</span>
                                })}
                            </button>
                        }
                    }).collect::<Vec<_>>()}

                    // Calendar link
                    <a href="/calendar" class="folder-item">
                        <span class="folder-icon">"\u{1F4C5}"</span>
                        <span class="folder-name">"Calendar"</span>
                    </a>
                </div>
            </div>

            // ── THE PULSE (real-time, ephemeral signals) ──
            <div class="sidebar-section pulse-section">
                <h4 class="sidebar-section-title pulse-title">"The Pulse"</h4>

                // Direct messages
                {move || {
                    let dms: Vec<_> = chat_channels.get().iter().filter(|c| c.is_direct).cloned().collect();
                    if dms.is_empty() { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="pulse-group">
                            {dms.into_iter().map(|ch| {
                                let name = ch.name.clone();
                                let unread = ch.unread_count;
                                let id = ch.id.clone();
                                view! {
                                    <a href=format!("/chat?ch={id}") class="folder-item pulse-item">
                                        <span class="pulse-dot presence-online" />
                                        <span class="folder-name">{name}</span>
                                        {(unread > 0).then(|| view! { <span class="folder-badge pulse-badge">{unread}</span> })}
                                    </a>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}

                // Channels
                {move || {
                    let channels: Vec<_> = chat_channels.get().iter().filter(|c| !c.is_direct).cloned().collect();
                    if channels.is_empty() { return view! { <div /> }.into_any(); }
                    view! {
                        <div class="pulse-group">
                            {channels.into_iter().map(|ch| {
                                let name = ch.name.clone();
                                let unread = ch.unread_count;
                                let id = ch.id.clone();
                                view! {
                                    <a href=format!("/chat?ch={id}") class="folder-item pulse-item">
                                        <span class="channel-prefix">"#"</span>
                                        <span class="folder-name">{name}</span>
                                        {(unread > 0).then(|| view! { <span class="folder-badge pulse-badge">{unread}</span> })}
                                    </a>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}

                // Meet
                <a href="/meet" class="folder-item pulse-item meet-link">
                    <span class="folder-icon">"\u{1F4F9}"</span>
                    <span class="folder-name">"Start Call"</span>
                </a>
            </div>

            // ── TRUST DASHBOARD ──
            <div class="sidebar-section trust-section">
                <h4 class="sidebar-section-title">"\u{1F6E1} Trust Network"</h4>
                <div class="trust-summary">
                    {
                        let mail_trust = mail_trust_sidebar.clone();
                        move || {
                        let trust_map = mail_trust.sender_trust.get();
                        let total = trust_map.len();
                        let trusted = trust_map.values().filter(|s| **s >= 0.8).count();
                        let staked = trust_map.values().filter(|s| **s >= 0.5 && **s < 0.8).count();
                        let quarantined = trust_map.values().filter(|s| **s < 0.5).count();
                        let avg_trust = if total > 0 {
                            trust_map.values().sum::<f64>() / total as f64
                        } else { 0.0 };
                        view! {
                            <div class="trust-stats">
                                <div class="trust-stat">
                                    <span class="trust-stat-value trust-high">{trusted}</span>
                                    <span class="trust-stat-label">"Trusted"</span>
                                </div>
                                <div class="trust-stat">
                                    <span class="trust-stat-value trust-medium">{staked}</span>
                                    <span class="trust-stat-label">"Staked"</span>
                                </div>
                                <div class="trust-stat">
                                    <span class="trust-stat-value trust-low">{quarantined}</span>
                                    <span class="trust-stat-label">"Guarded"</span>
                                </div>
                            </div>
                            <div class="trust-avg">
                                <span class="trust-avg-label">"Network Trust"</span>
                                <div class="trust-bar">
                                    <div class="trust-bar-fill" style=format!("width:{}%", avg_trust * 100.0) />
                                </div>
                                <span class="trust-avg-value">{format!("{:.0}%", avg_trust * 100.0)}</span>
                            </div>
                        }
                    }}
                </div>
            </div>

            // ── ATTENTION BUDGET ──
            <div class="sidebar-section attention-section">
                <h4 class="sidebar-section-title">"\u{1F9E0} Attention"</h4>
                {
                    let mail_att = mail_att_sidebar.clone();
                    move || {
                    let inbox_count = mail_att.inbox.get().len();
                    let unread = mail_att.inbox.get().iter().filter(|e| !e.is_read).count();
                    // Simulate attention budget usage (in prod, tracked via time-on-page)
                    let budget = 100u32;
                    let used = (inbox_count.min(budget as usize) as f64 / budget as f64 * 100.0) as u32;
                    let bar_color = if used > 80 { "var(--error)" } else if used > 50 { "var(--warning, #f59e0b)" } else { "var(--primary)" };
                    view! {
                        <div class="attention-budget">
                            <div class="attention-bar-container">
                                <div class="attention-bar-fill" style=format!("width:{used}%; background:{bar_color}") />
                            </div>
                            <div class="attention-stats">
                                <span class="attention-stat">{format!("{unread} unread")}</span>
                                <span class="attention-stat">{format!("{used}% budget")}</span>
                            </div>
                        </div>
                    }
                }}
            </div>

            <div class="sidebar-footer">
                // User status
                <UserStatus />
                <div class="storage-info">
                    <span class="storage-label">"DHT Storage"</span>
                    <span class="storage-value">"Distributed"</span>
                </div>
                <div class="keyboard-hint">
                    <span class="hint-text">"Press ? for shortcuts"</span>
                </div>
            </div>
        </aside>
    }
}

/// User status picker — set custom status with auto-expire.
#[component]
fn UserStatus() -> impl IntoView {
    let status_text = RwSignal::new(
        web_sys::window().and_then(|w| w.local_storage().ok().flatten())
            .and_then(|s| s.get_item("mycelix_user_status").ok().flatten())
            .unwrap_or_default()
    );
    let open = RwSignal::new(false);

    let presets = [
        ("\u{1F3AF}", "Focusing"),
        ("\u{1F4BC}", "In a meeting"),
        ("\u{1F3E0}", "Working from home"),
        ("\u{1F30D}", "Away"),
        ("\u{1F3B5}", "Listening to music"),
        ("", "Clear status"),
    ];

    view! {
        <div class="user-status-area">
            <button class="status-trigger" on:click=move |_| open.update(|v| *v = !*v)>
                {move || {
                    let s = status_text.get();
                    if s.is_empty() { "\u{1F7E2} Set status".to_string() } else { s }
                }}
            </button>
            <div class="status-picker" style=move || if open.get() { "" } else { "display:none" }>
                {presets.iter().map(|(emoji, label)| {
                    let emoji = *emoji;
                    let label = *label;
                    view! {
                        <button class="status-preset" on:click=move |_| {
                            let text = if emoji.is_empty() { String::new() } else { format!("{emoji} {label}") };
                            status_text.set(text.clone());
                            if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                                if text.is_empty() { let _ = s.remove_item("mycelix_user_status"); }
                                else { let _ = s.set_item("mycelix_user_status", &text); }
                            }
                            open.set(false);
                        }>{emoji}" "{label}</button>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
