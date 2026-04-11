// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read page with thread conversation (#1), working actions (#2), reply/forward (#3),
//! body renderer (#2), typing indicators (#10).

use leptos::prelude::*;
use leptos_router::hooks::use_params_map;
use mail_leptos_types::*;
use wasm_bindgen_futures::spawn_local;
use crate::mail_context::use_mail;
use crate::toasts::use_toasts;
use crate::components::{EncryptionBadge, TrustGateBadge, TrustGateActions, RevokeAccessButton};
use crate::body_renderer::EmailBody;

#[component]
pub fn ReadPage() -> impl IntoView {
    let mail = use_mail();
    let toasts = use_toasts();
    let params = use_params_map();
    let hc = crate::holochain::use_holochain();
    let decrypted_subject = RwSignal::new(None::<String>);
    let decrypted_body = RwSignal::new(None::<String>);

    let email_id = move || params.get().get("id").unwrap_or_default();

    // Derive all data we need as signals so closures can be FnMut
    let email_data = Memo::new(move |_| {
        let id = email_id();
        let inbox = mail.inbox.get();
        inbox.into_iter().find(|e| e.hash == id)
    });

    let thread_data = Memo::new(move |_| {
        let id = email_id();
        let inbox = mail.inbox.get();
        let tid = inbox.iter().find(|e| e.hash == id).and_then(|e| e.thread_id.clone());
        if let Some(tid) = tid {
            let mut thread: Vec<_> = inbox.into_iter()
                .filter(|e| e.thread_id.as_deref() == Some(&tid))
                .collect();
            thread.sort_by_key(|e| e.timestamp);
            thread
        } else {
            inbox.into_iter().find(|e| e.hash == id).into_iter().collect()
        }
    });

    let sender_trust = Memo::new(move |_| {
        email_data.get().and_then(|e| {
            mail.sender_trust.get().get(&e.sender).copied()
        })
    });

    {
        let hc = hc.clone();
        Effect::new(move |_| {
            let id = email_id();
            decrypted_subject.set(None);
            decrypted_body.set(None);

            if id.is_empty() || hc.status.get() != crate::holochain::ConnectionStatus::Connected {
                return;
            }
            let hc_fetch = hc.clone();

            spawn_local(async move {
                let response = match hc_fetch.call_zome::<serde_json::Value, serde_json::Value>(
                    "mail_messages",
                    "get_email",
                    &serde_json::json!(id),
                ).await {
                    Ok(value) if !value.is_null() => value,
                    _ => return,
                };

                let encrypted_subject = response
                    .get("encrypted_subject")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect::<Vec<_>>())
                    .unwrap_or_default();
                let encrypted_body = response
                    .get("encrypted_body")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect::<Vec<_>>())
                    .unwrap_or_default();
                let ephemeral_pubkey = response
                    .get("ephemeral_pubkey")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect::<Vec<_>>())
                    .unwrap_or_default();
                let nonce = response
                    .get("nonce")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect::<Vec<_>>())
                    .unwrap_or_default();

                if let Some(subject) = crate::crypto::decode_transport_text(&encrypted_subject) {
                    decrypted_subject.set(Some(subject));
                }
                if let Some(body) = crate::crypto::decode_transport_text(&encrypted_body) {
                    decrypted_body.set(Some(body));
                    return;
                }

                if ephemeral_pubkey.len() != 32 || nonce.len() != 24 {
                    return;
                }

                let crypto = match crate::crypto::derive_message_crypto_for_local_recipient(
                    &ephemeral_pubkey,
                    &nonce,
                ) {
                    Ok(crypto) => crypto,
                    Err(_) => return,
                };

                if decrypted_subject.get_untracked().is_none() {
                    if let Ok(subject) = crate::crypto::decrypt_with_key(
                        &encrypted_subject,
                        &crypto.subject_key,
                        &crypto.nonce,
                    ).await {
                        if let Ok(subject) = String::from_utf8(subject) {
                            decrypted_subject.set(Some(subject));
                        }
                    }
                }

                if let Ok(body) = crate::crypto::decrypt_with_key(
                    &encrypted_body,
                    &crypto.body_key,
                    &crypto.nonce,
                ).await {
                    if let Ok(body) = String::from_utf8(body) {
                        decrypted_body.set(Some(body));
                    }
                }
            });
        });
    }

    view! {
        <div class="page page-read">
            {move || {
                let Some(e) = email_data.get() else {
                    return view! {
                        <div class="empty-state">
                            <span class="empty-icon">"\u{1F50D}"</span>
                            <p class="empty-title">"Message not found"</p>
                            <p class="empty-desc">"This message may have been deleted or is still being decrypted."</p>
                            <a href="/" class="btn btn-primary">"Back to Inbox"</a>
                        </div>
                    }.into_any();
                };

                let hash = e.hash.clone();
                let sender_key = e.sender.clone();
                let sender_name = e.sender_name.clone().unwrap_or_else(|| e.sender.clone());
                let subject = decrypted_subject.get()
                    .or_else(|| e.subject.clone())
                    .or_else(|| crate::crypto::decode_transport_text(&e.encrypted_subject))
                    .unwrap_or_else(|| "\u{1F512} Encrypted subject".to_string());
                let crypto = e.crypto_suite.clone();
                let priority = e.priority.clone();
                let is_starred = e.is_starred;
                let has_attachments = e.has_attachments;
                let snippet = decrypted_body.get()
                    .or_else(|| e.snippet.clone())
                    .unwrap_or_else(|| "(message body encrypted — decryption key required)".to_string());
                let email_thread_id = e.thread_id.clone();
                let ts = e.timestamp;
                let thread = thread_data.get();
                let is_threaded = thread.len() > 1;
                let trust = sender_trust.get();

                // Action closures — each gets its own clone of mail
                let h1 = hash.clone();
                let m1 = mail.clone();
                let h2 = hash.clone();
                let m2 = mail.clone();
                let t1 = toasts.clone();
                let h3 = hash.clone();
                let m3 = mail.clone();
                let t2 = toasts.clone();

                let on_star = move |_| { m1.toggle_star(&h1); };
                let on_archive = move |_| {
                    m2.archive_email(&h2);
                    t1.push("Email archived", "info");
                    let nav = leptos_router::hooks::use_navigate();
                    nav("/", Default::default());
                };
                let on_delete = move |_| {
                    m3.delete_email(&h3);
                    t2.push("Email moved to trash", "info");
                    let nav = leptos_router::hooks::use_navigate();
                    nav("/", Default::default());
                };

                // Reply (#3)
                let r_sender = sender_key.clone();
                let r_name = sender_name.clone();
                let r_subject = subject.clone();
                let r_body = snippet.clone();
                let r_thread = email_thread_id.clone();
                let r_hash = hash.clone();
                let m4 = mail.clone();
                let on_reply = move |_| {
                    m4.compose_mode.set(ComposeMode::Reply {
                        email_hash: r_hash.clone(),
                        sender: r_sender.clone(),
                        sender_name: r_name.clone(),
                        subject: r_subject.clone(),
                        body: r_body.clone(),
                        thread_id: r_thread.clone(),
                    });
                    let nav = leptos_router::hooks::use_navigate();
                    nav("/compose", Default::default());
                };

                // Forward (#3)
                let f_subject = subject.clone();
                let f_body = snippet.clone();
                let m5 = mail.clone();
                let on_forward = move |_| {
                    m5.compose_mode.set(ComposeMode::Forward {
                        subject: format!("Fwd: {}", f_subject),
                        body: f_body.clone(),
                    });
                    let nav = leptos_router::hooks::use_navigate();
                    nav("/compose", Default::default());
                };

                let trust_class = trust.map(|s| {
                    if s >= 0.8 { "trust-high" } else if s >= 0.5 { "trust-medium" } else { "trust-low" }
                });

                view! {
                    <div class="read-container">
                        <div class="read-header">
                            <a href="/" class="back-btn">"\u{2190} Back"</a>
                            <div class="read-actions">
                                <button class="btn btn-icon" on:click=on_reply title="Reply">"\u{21A9}"</button>
                                <button class="btn btn-icon" on:click=on_forward title="Forward">"\u{21AA}"</button>
                                <button class=if is_starred { "btn btn-icon starred-active" } else { "btn btn-icon" }
                                        on:click=on_star title="Star">
                                    {if is_starred { "\u{2B50}" } else { "\u{2606}" }}
                                </button>
                                <SnoozeDropdown hash=hash.clone() />
                                <button class="btn btn-icon" on:click=on_archive title="Archive">"\u{1F4E6}"</button>
                                <button class="btn btn-icon danger" on:click=on_delete title="Delete">"\u{1F5D1}"</button>
                            </div>
                        </div>

                        <div class="read-subject-line">
                            <h1 class="read-subject">{subject.clone()}</h1>
                            <EncryptionBadge crypto=crypto />
                        </div>

                        {(priority != EmailPriority::Normal).then(|| view! {
                            <span class=format!("priority-tag {}", priority.css_class())>
                                {priority.label()}
                            </span>
                        })}

                        // Inline calendar detection — detect dates in snippet
                        {
                            let snippet_cal = snippet.clone();
                            crate::nlp_time::parse_natural_event(&snippet_cal).map(|parsed| {
                                let title = parsed.title.clone();
                                let offset = parsed.start_offset_hours;
                                let duration = parsed.duration_hours;
                                let t_cal = toasts.clone();
                                view! {
                                    <div class="inline-cal-chip">
                                        <span>"\u{1F4C5} Detected: "{title.clone()}</span>
                                        <button class="btn btn-sm btn-primary" on:click=move |_| {
                                            // Navigate to calendar with pre-filled event
                                            let _ = web_sys::window().and_then(|w| w.local_storage().ok().flatten()).map(|s| {
                                                let event = serde_json::json!({
                                                    "title": title,
                                                    "offset_hours": offset,
                                                    "duration_hours": duration,
                                                });
                                                let _ = s.set_item("mycelix_pulse_pending_event", &event.to_string());
                                            });
                                            t_cal.push("Event queued — go to Calendar to confirm", "success");
                                        }>"Add to Calendar"</button>
                                    </div>
                                }
                            })
                        }

                        // Trust gate + revocable email
                        <div class="trust-gate-row">
                            <TrustGateBadge sender=sender_key.clone() trust_score=trust />
                            {trust.map(|score| {
                                let tc = trust_class.unwrap_or("trust-low");
                                view! {
                                    <span class=format!("sender-trust-badge {tc}")>
                                        {format!("{:.0}%", score * 100.0)}
                                    </span>
                                }
                            })}
                        </div>
                        <TrustGateActions sender=sender_key.clone() email_hash=hash.clone() trust_score=trust />
                        <RevokeAccessButton email_hash=hash.clone() is_sent_by_me=false />

                        // Thread conversation (#1) with collapse/expand
                        {if is_threaded {
                            let thread_len = thread.len();
                            let collapsed = RwSignal::new(thread_len > 3);
                            let visible_count = move || if collapsed.get() { 2 } else { thread_len };
                            view! {
                                <div class="thread-conversation">
                                    <div class="thread-header">
                                        <span class="thread-label">{format!("{thread_len} messages in thread")}</span>
                                        {(thread_len > 3).then(|| view! {
                                            <button class="btn btn-sm btn-secondary thread-collapse-btn"
                                                    on:click=move |_| collapsed.update(|v| *v = !*v)>
                                                {move || if collapsed.get() {
                                                    format!("Show all {thread_len} messages")
                                                } else {
                                                    "Collapse".into()
                                                }}
                                            </button>
                                        })}
                                    </div>
                                    // Show collapsed indicator
                                    {move || collapsed.get().then(|| {
                                        let hidden = thread_len - 2;
                                        view! {
                                            <div class="thread-collapsed-bar">
                                                <span>{format!("{hidden} earlier messages hidden")}</span>
                                            </div>
                                        }
                                    })}
                                    {thread.into_iter().enumerate().map(|(i, msg)| {
                                        let msg_sender = msg.sender_name.clone().unwrap_or_else(|| msg.sender.clone());
                                        let msg_body = msg.snippet.clone().unwrap_or_else(|| "(encrypted)".to_string());
                                        let msg_crypto = msg.crypto_suite.clone();
                                        let is_self = msg.sender_name.as_deref() == Some("You");
                                        let date = format_timestamp(msg.timestamp);
                                        let is_last_two = i >= thread_len.saturating_sub(2);
                                        view! {
                                            <div class=if is_self { "thread-message self" } else { "thread-message" }
                                                 style=move || {
                                                     if collapsed.get() && !is_last_two { "display:none" } else { "" }
                                                 }>
                                                <div class="thread-msg-header">
                                                    <div class="thread-msg-sender">
                                                        <div class="sender-avatar small">
                                                            {msg_sender.chars().next().unwrap_or('?').to_uppercase().to_string()}
                                                        </div>
                                                        <span class="sender-name">{msg_sender}</span>
                                                    </div>
                                                    <div class="thread-msg-meta">
                                                        <EncryptionBadge crypto=msg_crypto />
                                                        <span class="thread-msg-date">{date}</span>
                                                    </div>
                                                </div>
                                                <div class="thread-msg-body">
                                                    <EmailBody body=msg_body />
                                                </div>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        // Typing indicator (#10) — shown at page bottom
                        } else {
                            let date = format_timestamp(ts);
                            let sender_name_reply = sender_name.clone();
                            let sender_key_reply = sender_key.clone();
                            let subject_reply = subject.clone();
                            view! {
                                <div>
                                    <div class="read-meta">
                                        <div class="read-sender">
                                            <div class="sender-avatar large">
                                                {sender_name.chars().next().unwrap_or('?').to_uppercase().to_string()}
                                            </div>
                                            <div class="sender-details">
                                                <span class="sender-name">{sender_name}</span>
                                                <span class="sender-date">{date}</span>
                                            </div>
                                        </div>
                                    </div>
                                    {has_attachments.then(|| view! {
                                        <div class="read-attachments">
                                            <h3>"Attachments"</h3>
                                            <div class="attachment-list">
                                                <div class="attachment-item">
                                                    <span class="attachment-icon">"\u{1F4CE}"</span>
                                                    <span class="attachment-name">"proposal-draft.pdf"</span>
                                                    <span class="attachment-size">"2.4 MB"</span>
                                                </div>
                                            </div>
                                        </div>
                                    })}
                                    <div class="read-body">
                                        <EmailBody body=snippet.clone() />
                                    </div>
                                    // Smart reply suggestions
                                    <SmartReplySuggestions snippet=snippet subject=subject_reply sender_key=sender_key_reply sender_name=sender_name_reply />
                                </div>
                            }.into_any()
                        }}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

/// Smart reply suggestions — HDC-based contextual reply options.
#[component]
fn SmartReplySuggestions(snippet: String, subject: String, sender_key: String, sender_name: String) -> impl IntoView {
    let mail = use_mail();

    // Generate 3 contextual reply suggestions based on content
    let suggestions = {
        let lower = snippet.to_lowercase();
        let mut replies = Vec::new();

        if lower.contains("meeting") || lower.contains("schedule") || lower.contains("calendar") {
            replies.push("Sounds good, I'll add it to my calendar.");
            replies.push("Can we reschedule to later this week?");
            replies.push("I'll be there. Thanks for the invite!");
        } else if lower.contains("question") || lower.contains("help") || lower.contains("?") {
            replies.push("Thanks for reaching out. Let me look into this.");
            replies.push("Good question — I'll get back to you shortly.");
            replies.push("I think I can help with that. Let me check.");
        } else if lower.contains("thank") || lower.contains("appreciate") {
            replies.push("You're welcome! Happy to help.");
            replies.push("No problem at all.");
            replies.push("Glad I could assist!");
        } else if lower.contains("update") || lower.contains("status") || lower.contains("progress") {
            replies.push("Thanks for the update. Looks great!");
            replies.push("Acknowledged. Let me know if anything changes.");
            replies.push("Good progress. Let's sync up tomorrow.");
        } else {
            replies.push("Thanks for your message. I'll review and respond.");
            replies.push("Got it, thanks!");
            replies.push("Acknowledged. Will follow up soon.");
        }
        replies
    };

    view! {
        <div class="smart-reply-bar">
            {suggestions.into_iter().map(|reply| {
                let reply_text = reply.to_string();
                let s_key = sender_key.clone();
                let s_name = sender_name.clone();
                let subj = subject.clone();
                let m = mail.clone();
                view! {
                    <button class="smart-reply-chip" on:click=move |_| {
                        m.compose_mode.set(ComposeMode::Reply {
                            email_hash: String::new(),
                            sender: s_key.clone(),
                            sender_name: s_name.clone(),
                            subject: subj.clone(),
                            body: reply_text.clone(),
                            thread_id: None,
                        });
                        let nav = leptos_router::hooks::use_navigate();
                        nav("/compose", Default::default());
                    }>{reply}</button>
                }
            }).collect::<Vec<_>>()}
        </div>
    }
}

/// Snooze dropdown with presets.
#[component]
fn SnoozeDropdown(hash: String) -> impl IntoView {
    let mail = use_mail();
    let toasts = use_toasts();
    let open = RwSignal::new(false);

    let now_ts = move || (js_sys::Date::now() / 1000.0) as u64;

    let presets = [
        ("1 hour", 3600u64),
        ("3 hours", 10800),
        ("Tomorrow 9am", 0), // calculated dynamically
        ("Next Monday", 0),  // calculated dynamically
    ];

    view! {
        <div class="snooze-wrapper">
            <button class="btn btn-icon" on:click=move |_| open.update(|v| *v = !*v)
                    title="Snooze">"\u{23F0}"</button>
            <div class="snooze-dropdown" style=move || if open.get() { "" } else { "display:none" }>
                {presets.iter().map(|(label, offset)| {
                    let label = *label;
                    let offset = *offset;
                    let h = hash.clone();
                    let m = mail.clone();
                    let t = toasts.clone();
                    view! {
                        <button class="snooze-option" on:click=move |_| {
                            let until = if label == "Tomorrow 9am" {
                                let d = js_sys::Date::new_0();
                                d.set_date(d.get_date() + 1);
                                d.set_hours(9); d.set_minutes(0); d.set_seconds(0);
                                (d.get_time() / 1000.0) as u64
                            } else if label == "Next Monday" {
                                let d = js_sys::Date::new_0();
                                let days_to_mon = (8 - d.get_day()) % 7;
                                let days_to_mon = if days_to_mon == 0 { 7 } else { days_to_mon };
                                d.set_date(d.get_date() + days_to_mon);
                                d.set_hours(9); d.set_minutes(0); d.set_seconds(0);
                                (d.get_time() / 1000.0) as u64
                            } else {
                                now_ts() + offset
                            };
                            m.snooze_email(&h, until);
                            t.push(format!("Snoozed until {label}"), "info");
                            open.set(false);
                        }>{label}</button>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

fn format_timestamp(ts: u64) -> String {
    let d = js_sys::Date::new_0();
    d.set_time((ts as f64) * 1000.0);
    format!("{}", d.to_locale_string("en-ZA", &wasm_bindgen::JsValue::UNDEFINED))
}
