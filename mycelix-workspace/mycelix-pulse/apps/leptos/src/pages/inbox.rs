// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Inbox with pinned conversations, date separators, split view, thread grouping, batch actions.

use leptos::prelude::*;
use crate::mail_context::use_mail;
use crate::holochain::{use_holochain, ConnectionStatus};
use crate::toasts::use_toasts;
use crate::components::{EmailCard, NotificationPrompt, PullToRefresh};
use crate::semantic;

#[component]
pub fn InboxPage() -> impl IntoView {
    let mail = use_mail();
    let _toasts = use_toasts();
    let view_threaded = RwSignal::new(true);
    let view_clustered = RwSignal::new(false);
    let sort_priority = RwSignal::new(false);

    let folder_name = move || mail.active_folder.get();

    let mail_emails = mail.clone();
    let all_emails = move || {
        let folder = mail_emails.active_folder.get();
        match folder.as_str() {
            "Starred" => {
                let mut emails = mail_emails.inbox.get();
                emails.retain(|e| e.is_starred);
                emails
            }
            "Sent" => mail_emails.sent.get(),
            _ => mail_emails.inbox.get(),
        }
    };

    // Split into pinned and unpinned
    let pinned = move || {
        all_emails().into_iter().filter(|e| e.is_pinned).collect::<Vec<_>>()
    };
    let mail_priority = mail.clone();
    let unpinned = move || {
        let mut emails: Vec<_> = all_emails().into_iter().filter(|e| !e.is_pinned).collect();
        if sort_priority.get() {
            // Priority score: trust * 30 + unread * 25 + starred * 20 + recency * 15 + thread * 10
            let trust_map = mail_priority.sender_trust.get();
            emails.sort_by(|a, b| {
                let score = |e: &mail_leptos_types::EmailListItem| -> i64 {
                    let trust = trust_map.get(&e.sender).copied().unwrap_or(0.5);
                    let now = js_sys::Date::now() as u64 / 1000;
                    let age_hours = now.saturating_sub(e.timestamp) / 3600;
                    let recency = 100u64.saturating_sub(age_hours.min(100));

                    (trust * 30.0) as i64
                        + if !e.is_read { 25 } else { 0 }
                        + if e.is_starred { 20 } else { 0 }
                        + (recency as i64 * 15 / 100)
                        + if e.thread_id.is_some() { 10 } else { 0 }
                };
                score(b).cmp(&score(a))
            });
        }
        emails
    };

    let mail_threads = mail.clone();
    let threads = move || mail_threads.threaded_emails();
    let email_count = move || all_emails().len();
    let unread_count = move || all_emails().iter().filter(|e| !e.is_read).count();
    let _selection_count = move || mail.selected_hashes.get().len();

    // Split view state
    let split_email = RwSignal::new(Option::<String>::None);
    // Quick peek state (hover preview)
    let peek_email = RwSignal::new(Option::<(String, i32, i32)>::None); // (hash, x, y)
    let show_split = move || mail.reading_pane.get() != mail_leptos_types::ReadingPanePosition::Off && split_email.get().is_some();
    let split_data = move || {
        let id = split_email.get()?;
        mail.inbox.get().into_iter().find(|e| e.hash == id)
    };

    view! {
        <div class="page page-inbox">
            <div class="page-header">
                <h1>{folder_name}</h1>
                <div class="header-meta">
                    <span class="email-count">{move || format!("{} messages", email_count())}</span>
                    {move || {
                        let unread = unread_count();
                        (unread > 0).then(|| view! { <span class="unread-count">{format!("{unread} unread")}</span> })
                    }}
                    <button class="thread-toggle" on:click=move |_| view_threaded.update(|v| *v = !*v)
                            title="Toggle thread view">
                        {move || if view_threaded.get() { "Threaded" } else { "Flat" }}
                    </button>
                    <button class="thread-toggle cluster-toggle" on:click=move |_| view_clustered.update(|v| *v = !*v)
                            title="Toggle semantic clusters">
                        {move || if view_clustered.get() { "\u{1F9E0} Clustered" } else { "\u{1F9E0} Cluster" }}
                    </button>
                    <button class="thread-toggle priority-toggle" on:click=move |_| sort_priority.update(|v| *v = !*v)
                            title="Sort by priority score">
                        {move || if sort_priority.get() { "\u{26A1} Priority" } else { "\u{26A1} Priority" }}
                    </button>
                </div>
            </div>

            // Daily digest summary
            <EmailDigest />

            // Conversation heatmap (collapsible)
            {
                let show_heatmap = RwSignal::new(false);
                view! {
                    <div>
                        <button class="btn btn-sm btn-secondary heatmap-toggle"
                                on:click=move |_| show_heatmap.update(|v| *v = !*v)>
                            {move || if show_heatmap.get() { "\u{1F4CA} Hide Heatmap" } else { "\u{1F4CA} Activity Heatmap" }}
                        </button>
                        <div style=move || if show_heatmap.get() { "" } else { "display:none" }>
                            <crate::components::ConversationHeatmap />
                        </div>
                    </div>
                }
            }

            // Loading skeletons (shown while data loads)
            {move || mail.loading.get().then(|| view! {
                <div class="skeleton-list">
                    {(0..6).map(|_| view! {
                        <div class="skeleton-row">
                            <div class="skeleton skeleton-avatar" />
                            <div class="skeleton-lines">
                                <div class="skeleton skeleton-line short" />
                                <div class="skeleton skeleton-line long" />
                                <div class="skeleton skeleton-line medium" />
                            </div>
                            <div class="skeleton skeleton-actions" />
                        </div>
                    }).collect::<Vec<_>>()}
                </div>
            })}

            // Connection status banner
            <ConnectionBanner />
            <NotificationPrompt />

            <BatchToolbar />

            <div class=move || {
                match mail.reading_pane.get() {
                    mail_leptos_types::ReadingPanePosition::Right => "inbox-split-layout split-right",
                    mail_leptos_types::ReadingPanePosition::Bottom => "inbox-split-layout split-bottom",
                    _ => "inbox-split-layout",
                }
            }>
                // Email list pane
                <PullToRefresh on_refresh=move || {
                    mail.versions.inbox.update(|v| *v += 1);
                }>
                <div class="inbox-list-pane">
                    // Pinned section
                    {move || {
                        let pins = pinned();
                        (!pins.is_empty()).then(|| view! {
                            <div class="pinned-section">
                                <div class="section-label">"\u{1F4CC} Pinned"</div>
                                <div class="email-list pinned-list" role="list">
                                    {pins.into_iter().enumerate().map(|(idx, email)| {
                                        view! { <EmailCard email=email index=idx /> }
                                    }).collect::<Vec<_>>()}
                                </div>
                            </div>
                        })
                    }}

                    // Main email list — clustered, threaded, or flat
                    {move || {
                        // Semantic clustering mode
                        if view_clustered.get() {
                            let emails = unpinned();
                            if emails.is_empty() {
                                return empty_inbox_view().into_any();
                            }
                            // Encode emails into HDC vectors and cluster by similarity
                            let texts: Vec<String> = emails.iter().map(|e| {
                                format!("{} {}",
                                    e.subject.as_deref().unwrap_or(""),
                                    e.snippet.as_deref().unwrap_or(""))
                            }).collect();
                            let hvs: Vec<semantic::HyperVector> = texts.iter()
                                .map(|t| semantic::encode_text(t)).collect();

                            // Simple greedy clustering: assign each email to closest cluster
                            let mut clusters: Vec<(String, Vec<usize>)> = Vec::new();
                            for (i, hv) in hvs.iter().enumerate() {
                                let mut best_cluster = None;
                                let mut best_sim = 0.15f32; // threshold
                                for (ci, (_, members)) in clusters.iter().enumerate() {
                                    let centroid = &hvs[members[0]];
                                    let sim = hv.similarity(centroid);
                                    if sim > best_sim {
                                        best_sim = sim;
                                        best_cluster = Some(ci);
                                    }
                                }
                                if let Some(ci) = best_cluster {
                                    clusters[ci].1.push(i);
                                } else {
                                    // New cluster — name from first email's subject
                                    let name = emails[i].subject.clone().unwrap_or_else(|| "Other".into());
                                    // Truncate cluster name
                                    let label = if name.len() > 30 { format!("{}...", &name[..27]) } else { name };
                                    clusters.push((label, vec![i]));
                                }
                            }

                            return view! {
                                <div class="clustered-inbox">
                                    {clusters.into_iter().map(|(label, indices)| {
                                        let count = indices.len();
                                        view! {
                                            <div class="cluster-group">
                                                <div class="cluster-header">
                                                    <span class="cluster-icon">"\u{1F4A1}"</span>
                                                    <span class="cluster-label">{label}</span>
                                                    <span class="cluster-count">{format!("({count})")}</span>
                                                </div>
                                                <div class="email-list" role="list">
                                                    {indices.into_iter().filter_map(|i| emails.get(i).cloned()).enumerate().map(|(idx, email)| {
                                                        view! { <EmailCard email=email index=idx /> }
                                                    }).collect::<Vec<_>>()}
                                                </div>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any();
                        }

                        if view_threaded.get() && mail.active_folder.get() == "Inbox" {
                            let thread_list = threads();
                            if thread_list.is_empty() && pinned().is_empty() {
                                return empty_inbox_view().into_any();
                            }
                            view! {
                                <div class="thread-list" role="list">
                                    {thread_list.into_iter().enumerate().map(|(idx, thread)| {
                                        let latest = thread.messages.last().cloned();
                                        let msg_count = thread.messages.len();
                                        let has_multiple = msg_count > 1;
                                        if let Some(email) = latest {
                                            view! {
                                                <div class=if has_multiple { "thread-group multi" } else { "thread-group" }>
                                                    {has_multiple.then(|| {
                                                        let participants = thread.participant_summary();
                                                        view! {
                                                            <div class="thread-bar">
                                                                <span class="thread-participants">{participants}</span>
                                                                <span class="thread-msg-count">{format!("{msg_count} messages")}</span>
                                                                {(thread.unread_count > 0).then(|| view! {
                                                                    <span class="thread-unread">{format!("{} new", thread.unread_count)}</span>
                                                                })}
                                                            </div>
                                                        }
                                                    })}
                                                    <EmailCard email=email index=idx />
                                                </div>
                                            }.into_any()
                                        } else {
                                            view! { <div /> }.into_any()
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        } else {
                            let list = unpinned();
                            if list.is_empty() && pinned().is_empty() {
                                return empty_inbox_view().into_any();
                            }

                            // Group by date for date separators
                            let mut last_date = String::new();
                            let items: Vec<_> = list.into_iter().enumerate().map(|(idx, email)| {
                                let date = date_label(email.timestamp);
                                let show_separator = date != last_date;
                                last_date = date.clone();
                                (email, idx, date, show_separator)
                            }).collect();

                            view! {
                                <div class="email-list" role="list">
                                    {items.into_iter().map(|(email, idx, date, show_sep)| {
                                        view! {
                                            <div>
                                                {show_sep.then(|| view! {
                                                    <div class="date-separator">
                                                        <span class="date-label">{date}</span>
                                                    </div>
                                                })}
                                                <EmailCard email=email index=idx />
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
                </PullToRefresh>

                // Reading pane (split view)
                {move || show_split().then(|| {
                    let data = split_data();
                    view! {
                        <div class="reading-pane">
                            {match data {
                                Some(e) => {
                                    let sender = e.sender_name.clone().unwrap_or_else(|| e.sender.clone());
                                    let subject = e.subject.clone().unwrap_or_else(|| "(encrypted)".to_string());
                                    let body = e.snippet.clone().unwrap_or_else(|| "(encrypted body)".to_string());
                                    let crypto = e.crypto_suite.clone();
                                    view! {
                                        <div class="reading-pane-content">
                                            <div class="reading-pane-header">
                                                <h2>{subject}</h2>
                                                <button class="btn btn-icon" on:click=move |_| split_email.set(None)>"\u{2715}"</button>
                                            </div>
                                            <div class="reading-pane-meta">
                                                <span class="reading-sender">{sender}</span>
                                                <crate::components::EncryptionBadge crypto=crypto />
                                            </div>
                                            <div class="reading-pane-body">
                                                <crate::body_renderer::EmailBody body=body />
                                            </div>
                                        </div>
                                    }.into_any()
                                }
                                None => view! { <div class="reading-pane-empty">"Select an email"</div> }.into_any()
                            }}
                        </div>
                    }
                })}
            </div>
            // Quick peek overlay
            {move || peek_email.get().and_then(|(hash, x, y)| {
                let email = mail.inbox.get().into_iter().find(|e| e.hash == hash)?;
                let subject = email.subject.clone().unwrap_or_else(|| "(encrypted)".into());
                let sender = email.sender_name.clone().unwrap_or_default();
                let snippet = email.snippet.clone().unwrap_or_default();
                let snip_preview = if snippet.len() > 200 { format!("{}...", &snippet[..197]) } else { snippet };
                Some(view! {
                    <div class="email-peek" style=format!("top:{}px;left:{}px;", y, x.min(600))>
                        <div class="peek-subject">{subject}</div>
                        <div class="peek-sender">{sender}</div>
                        <div class="peek-snippet">{snip_preview}</div>
                    </div>
                })
            })}
        </div>
    }
}

/// Daily email digest — summary of activity.
#[component]
fn EmailDigest() -> impl IntoView {
    let mail = use_mail();
    let show = RwSignal::new(true);

    view! {
        <div class="email-digest" style=move || if show.get() { "" } else { "display:none" }>
            <div class="digest-header">
                <h3>"\u{1F4CA} Today's Digest"</h3>
                <button class="btn btn-icon digest-close" on:click=move |_| show.set(false)>"\u{2715}"</button>
            </div>
            {move || {
                let emails = mail.inbox.get();
                let now = js_sys::Date::now() as u64 / 1000;
                let today_start = now - (now % 86400);
                let today_emails: Vec<_> = emails.iter().filter(|e| e.timestamp >= today_start).collect();
                let unread_today = today_emails.iter().filter(|e| !e.is_read).count();

                // Group by sender
                let mut sender_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                for e in &today_emails {
                    let name = e.sender_name.clone().unwrap_or_else(|| "Unknown".into());
                    *sender_counts.entry(name).or_default() += 1;
                }
                let mut top_senders: Vec<_> = sender_counts.into_iter().collect();
                top_senders.sort_by(|a, b| b.1.cmp(&a.1));
                let top_senders: Vec<_> = top_senders.into_iter().take(3).collect();

                let trust_pending = emails.iter().filter(|e| {
                    mail.sender_trust.get().get(&e.sender).map(|s| *s < 0.5).unwrap_or(false)
                }).count();

                view! {
                    <div class="digest-stats">
                        <span class="digest-stat">{format!("{} new today", today_emails.len())}</span>
                        <span class="digest-stat">{format!("{unread_today} unread")}</span>
                        {(trust_pending > 0).then(|| view! {
                            <span class="digest-stat digest-trust">{format!("{trust_pending} trust reviews")}</span>
                        })}
                    </div>
                    {(!top_senders.is_empty()).then(|| {
                        let senders = top_senders.clone();
                        view! {
                            <div class="digest-senders">
                                {senders.into_iter().map(|(name, count)| {
                                    view! { <span class="digest-sender">{format!("{count} from {name}")}</span> }
                                }).collect::<Vec<_>>()}
                            </div>
                        }
                    })}
                }
            }}
        </div>
    }
}

#[component]
fn BatchToolbar() -> impl IntoView {
    let mail = use_mail();
    let toasts = use_toasts();
    let has_selection = move || !mail.selected_hashes.get().is_empty();
    let sel_count = move || mail.selected_hashes.get().len();

    let m1 = mail.clone(); let m2 = mail.clone(); let m3 = mail.clone();
    let m4 = mail.clone(); let m5 = mail.clone(); let m6 = mail.clone();
    let t1 = toasts.clone(); let t2 = toasts.clone();

    view! {
        <div class="batch-toolbar" style=move || if has_selection() { "display:flex" } else { "display:none" }>
            <span class="batch-count">{move || format!("{} selected", sel_count())}</span>
            <button class="btn btn-sm btn-secondary" on:click=move |_| m1.select_all()>"Select All"</button>
            <button class="btn btn-sm btn-secondary" on:click=move |_| m2.deselect_all()>"Deselect"</button>
            <div class="batch-divider" />
            <button class="btn btn-sm btn-secondary" on:click=move |_| m3.batch_mark_read()>"\u{2709} Read"</button>
            <button class="btn btn-sm btn-secondary" on:click=move |_| m4.batch_star()>"\u{2B50} Star"</button>
            <button class="btn btn-sm btn-secondary" on:click=move |_| { let c = m5.selected_hashes.get_untracked().len(); m5.batch_archive(); t1.push(format!("{c} archived"), "info"); }>"\u{1F4E6} Archive"</button>
            <button class="btn btn-sm btn-secondary batch-delete" on:click=move |_| { let c = m6.selected_hashes.get_untracked().len(); m6.batch_delete(); t2.push(format!("{c} deleted"), "info"); }>"\u{1F5D1} Delete"</button>
        </div>
    }
}

#[component]
fn ConnectionBanner() -> impl IntoView {
    let hc = use_holochain();
    let is_mock = move || hc.status.get() == ConnectionStatus::Mock;
    let is_connecting = move || hc.status.get() == ConnectionStatus::Connecting;
    let is_disconnected = move || hc.status.get() == ConnectionStatus::Disconnected;

    view! {
        <div class="connection-banner mock" style=move || if is_mock() { "" } else { "display:none" }>
            "\u{1F6A7} Running in mock mode — showing sample data. "
            <a href="/settings">"Configure conductor"</a>
        </div>
        <div class="connection-banner connecting" style=move || if is_connecting() { "" } else { "display:none" }>
            "Connecting to conductor..."
        </div>
        <div class="connection-banner disconnected" style=move || if is_disconnected() { "" } else { "display:none" }>
            "Disconnected from conductor"
        </div>
    }
}

fn empty_inbox_svg() -> impl leptos::IntoView {
    // Inline SVG — minimalist envelope with checkmark
    view! {
        <svg viewBox="0 0 120 100" width="120" height="100" class="empty-svg" xmlns="http://www.w3.org/2000/svg">
            <rect x="15" y="25" width="90" height="60" rx="6" fill="none" stroke="#06D6C8" stroke-width="2" opacity="0.3" />
            <path d="M15 31 L60 60 L105 31" fill="none" stroke="#06D6C8" stroke-width="2" opacity="0.4" />
            <circle cx="85" cy="30" r="16" fill="#08080c" stroke="#22c55e" stroke-width="2" />
            <path d="M78 30 L83 35 L93 25" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
    }
}

fn empty_inbox_view() -> impl leptos::IntoView {
    // Inbox zero gamification
    let streak = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("mycelix_inbox_zero_streak").ok().flatten())
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0) + 1;

    // Save updated streak
    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = s.set_item("mycelix_inbox_zero_streak", &streak.to_string());
    }

    let badges = [
        ("\u{2709}", "First Clear", streak >= 1),
        ("\u{1F525}", "3-Day Streak", streak >= 3),
        ("\u{1F31F}", "Week Warrior", streak >= 7),
        ("\u{1F3C6}", "Month Master", streak >= 30),
        ("\u{1F48E}", "Zen Master", streak >= 100),
    ];

    view! {
        <div class="inbox-zero-banner">
            {empty_inbox_svg()}
            <p class="empty-title">"Inbox Zero!"</p>
            <div class="streak-counter">{format!("\u{1F525} {streak} day streak")}</div>
            <p class="empty-desc">"You've cleared your inbox. Well done!"</p>
            <div class="badge-grid">
                {badges.iter().map(|(icon, label, earned)| {
                    let class = if *earned { "badge-item earned" } else { "badge-item" };
                    view! {
                        <div class=class title=*label>{*icon}</div>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

fn date_label(ts: u64) -> String {
    let now = js_sys::Date::new_0();
    let event = js_sys::Date::new_0();
    event.set_time((ts as f64) * 1000.0);

    let now_day = (now.get_time() / 86400000.0) as i64;
    let evt_day = (event.get_time() / 86400000.0) as i64;
    let diff = now_day - evt_day;

    if diff == 0 { "Today".into() }
    else if diff == 1 { "Yesterday".into() }
    else if diff < 7 { "This Week".into() }
    else if diff < 30 { "This Month".into() }
    else {
        let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        format!("{} {}", months[event.get_month() as usize], event.get_full_year())
    }
}
