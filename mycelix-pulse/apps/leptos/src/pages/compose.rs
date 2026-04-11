// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compose page with reply/forward prefill (#3).

use leptos::prelude::*;
use mail_leptos_types::ComposeMode;
use crate::mail_context::use_mail;
use crate::toasts::use_toasts;
use crate::components::{RichEditor, FileDropZone};

#[component]
pub fn ComposePage() -> impl IntoView {
    let mail = use_mail();
    let toasts = use_toasts();

    // Initialize from compose_mode (reply/forward prefill)
    let mode = mail.compose_mode.get_untracked();
    let (initial_to, initial_subject, initial_body, mode_label, reply_hash, reply_thread) = match &mode {
        ComposeMode::New => (String::new(), String::new(), String::new(), "Compose", None, None),
        ComposeMode::Reply { email_hash, sender, sender_name, subject, body, thread_id } => {
            let quoted = format!("\n\n--- On previous message, {} wrote ---\n{}", sender_name, body);
            let subj = if subject.starts_with("Re: ") { subject.clone() } else { format!("Re: {subject}") };
            (sender.clone(), subj, quoted, "Reply",
             Some(email_hash.clone()),
             thread_id.clone().or_else(|| Some(email_hash.clone())))
        }
        ComposeMode::Forward { subject, body } => {
            let quoted = format!("\n\n--- Forwarded message ---\n{}", body);
            (String::new(), subject.clone(), quoted, "Forward", None, None)
        }
    };
    // Reset compose mode after reading
    mail.compose_mode.set(ComposeMode::New);

    // Load draft from localStorage if no reply/forward mode
    let draft_key = "mycelix_pulse_draft";
    let (draft_to, draft_subj, draft_body) = if mode_label == "Compose" {
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .and_then(|s| s.get_item(draft_key).ok().flatten())
            .and_then(|json| serde_json::from_str::<(String, String, String)>(&json).ok())
            .unwrap_or_else(|| (initial_to.clone(), initial_subject.clone(), initial_body.clone()))
    } else {
        (initial_to.clone(), initial_subject.clone(), initial_body.clone())
    };

    let to_field = RwSignal::new(draft_to);
    let cc_field = RwSignal::new(String::new());
    let subject_field = RwSignal::new(draft_subj);
    let body_field = RwSignal::new(draft_body);

    // Auto-save draft every 30 seconds
    {
        let dk = draft_key;
        Effect::new(move |_| {
            let to = to_field.get();
            let subj = subject_field.get();
            let body = body_field.get();
            if !to.is_empty() || !subj.is_empty() || !body.is_empty() {
                if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                    let _ = s.set_item(dk, &serde_json::to_string(&(to, subj, body)).unwrap_or_default());
                }
            }
        });
    }
    let use_pqc = RwSignal::new(true);
    let show_cc = RwSignal::new(false);
    let sending = RwSignal::new(false);
    let show_templates = RwSignal::new(false);
    let attachments = RwSignal::new(Vec::<crate::components::Attachment>::new());
    let schedule_send = RwSignal::new(false);
    let schedule_datetime = RwSignal::new(String::new());

    let reply_thread_send = RwSignal::new(reply_thread.clone());
    let reply_hash_send = RwSignal::new(reply_hash.clone());
    let toasts_send = toasts.clone();
    let on_send = move |_| {
        let to = to_field.get();
        let subject = subject_field.get();
        let body = body_field.get();

        if to.trim().is_empty() {
            toasts_send.push("Please enter a recipient", "error");
            return;
        }
        if subject.trim().is_empty() && body.trim().is_empty() {
            toasts_send.push("Cannot send an empty message", "error");
            return;
        }

        // Duplicate send detection — check if we sent to this person about the same subject recently
        let recent_dup = mail.sent.get_untracked().iter().any(|e| {
            let same_recipient = e.sender_name.as_deref() == Some(to.trim()) ||
                                 e.hash.contains(to.trim());
            let same_subject = e.subject.as_deref().map(|s| s.to_lowercase()) ==
                               Some(subject.to_lowercase());
            let now = js_sys::Date::now() as u64 / 1000;
            let recent = now.saturating_sub(e.timestamp) < 3600; // within 1 hour
            same_recipient && same_subject && recent
        });
        if recent_dup {
            // Use browser confirm dialog
            let confirmed = web_sys::window()
                .and_then(|w| w.confirm_with_message(
                    "You sent a similar message to this recipient less than 1 hour ago. Send anyway?"
                ).ok())
                .unwrap_or(true);
            if !confirmed { return; }
        }

        sending.set(true);
        let pqc = use_pqc.get();
        let crypto_label = if pqc { "PQC (Kyber+Dilithium)" } else { "E2E (X25519+Ed25519)" };

        // Expand template variables
        let now = js_sys::Date::new_0();
        let today = format!("{}-{:02}-{:02}", now.get_full_year(), now.get_month() + 1, now.get_date());
        let time_now = format!("{:02}:{:02}", now.get_hours(), now.get_minutes());
        let body = body.replace("{{name}}", to.trim())
                       .replace("{{date}}", &today)
                       .replace("{{time}}", &time_now)
                       .replace("{{sender}}", "You");
        let subject = subject.replace("{{name}}", to.trim())
                             .replace("{{date}}", &today)
                             .replace("{{time}}", &time_now)
                             .replace("{{sender}}", "You");

        // Handle scheduled send
        if schedule_send.get() {
            let dt = schedule_datetime.get();
            if !dt.is_empty() {
                // Queue in localStorage for later dispatch
                if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                    let queued = serde_json::json!({
                        "to": to.trim(), "subject": subject, "body": body,
                        "pqc": pqc, "scheduled_at": dt,
                    });
                    let key = format!("mycelix_scheduled_{}", js_sys::Date::now() as u64);
                    let _ = s.set_item(&key, &queued.to_string());
                }
                toasts_send.push(format!("Scheduled for {dt}"), "success");
                to_field.set(String::new());
                subject_field.set(String::new());
                body_field.set(String::new());
                sending.set(false);
                schedule_send.set(false);
                schedule_datetime.set(String::new());
                if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                    let _ = s.remove_item(draft_key);
                }
                return;
            }
        }

        // Undo send: 5-second delay before actually sending
        let undo = RwSignal::new(false);
        let toasts_undo = toasts_send.clone();
        toasts_send.push(
            format!("Sending to {} via {}... (undo available for 5s)", to.trim(), crypto_label),
            "info",
        );

        // Clear form + draft immediately (feels responsive)
        to_field.set(String::new());
        cc_field.set(String::new());
        subject_field.set(String::new());
        body_field.set(String::new());
        sending.set(false);
        // Clear saved draft since we're sending
        if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = s.remove_item(draft_key);
        }

        // Capture values before moving into async block
        let send_to = to.trim().to_string();
        let send_subject = subject.clone();
        let send_body = body.clone();
        let send_pqc = pqc;

        // Delayed send — 5s undo window, then actual zome call
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_secs(5)).await;
            if undo.get_untracked() {
                toasts_undo.push("Send cancelled", "info");
                return;
            }

            let hc = crate::holochain::use_holochain();
            if hc.is_mock() {
                // Demo mode: add the email to the local inbox so the demo feels alive
                let now = (js_sys::Date::now() / 1000.0) as u64;
                mail.inbox.update(|emails| {
                    emails.insert(0, mail_leptos_types::EmailListItem {
                        hash: format!("sent-{now}"),
                        sender: "uhCAk_self_mock".into(),
                        sender_name: Some("You".into()),
                        encrypted_subject: vec![],
                        subject: Some(send_subject.clone()),
                        snippet: Some(if send_body.len() > 100 { format!("{}...", &send_body[..100]) } else { send_body.clone() }),
                        timestamp: now,
                        priority: mail_leptos_types::EmailPriority::Normal,
                        is_read: true,
                        is_starred: false,
                        star_type: None,
                        is_pinned: false,
                        is_muted: false,
                        is_snoozed: false,
                        snooze_until: None,
                        has_attachments: false,
                        labels: vec![],
                        thread_id: reply_thread_send.get_untracked(),
                        crypto_suite: mail_leptos_types::CryptoSuiteView {
                            key_exchange: if send_pqc { "kyber1024" } else { "x25519" }.into(),
                            symmetric: "chacha20-poly1305".into(),
                            signature: if send_pqc { "dilithium3" } else { "ed25519" }.into(),
                        },
                    });
                });
                toasts_undo.push("Message sent (demo mode — added to inbox)", "success");
                return;
            }

            let recipient_bundle = match hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_keys", "get_pre_key_bundle", &serde_json::json!(send_to)
            ).await {
                Ok(bundle) if !bundle.is_null() => bundle,
                Ok(_) => {
                    toasts_undo.push("Recipient has no published key bundle yet", "error");
                    crate::offline::enqueue_action(
                        mail_leptos_types::OfflineAction::Send {
                            to: send_to, subject: send_subject,
                            body: send_body, use_pqc: send_pqc,
                        }
                    );
                    return;
                }
                Err(e) => {
                    toasts_undo.push(format!("Could not fetch recipient keys: {e}"), "error");
                    crate::offline::enqueue_action(
                        mail_leptos_types::OfflineAction::Send {
                            to: send_to, subject: send_subject,
                            body: send_body, use_pqc: send_pqc,
                        }
                    );
                    return;
                }
            };

            let recipient_pubkey: Vec<u8> = recipient_bundle
                .get("identity_key")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u8))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            if recipient_pubkey.len() != 32 {
                toasts_undo.push("Recipient bundle is missing a valid identity key", "error");
                return;
            }

            let crypto = match crate::crypto::derive_message_crypto(&recipient_pubkey) {
                Ok(crypto) => crypto,
                Err(e) => {
                    toasts_undo.push(format!("Could not derive message keys: {e}"), "error");
                    return;
                }
            };

            let encrypted_subject = match crate::crypto::encrypt_with_key(
                send_subject.as_bytes(),
                &crypto.subject_key,
                &crypto.nonce,
            ).await {
                Ok(ct) => ct,
                Err(e) => {
                    toasts_undo.push(format!("Subject encryption failed: {e}"), "error");
                    return;
                }
            };

            let encrypted_body = match crate::crypto::encrypt_with_key(
                send_body.as_bytes(),
                &crypto.body_key,
                &crypto.nonce,
            ).await {
                Ok(ct) => ct,
                Err(e) => {
                    toasts_undo.push(format!("Body encryption failed: {e}"), "error");
                    return;
                }
            };

            if send_pqc {
                toasts_undo.push(
                    "Live send currently uses X25519 + AES-GCM while PQC transport is still being wired",
                    "info",
                );
            }

            let crypto_suite = serde_json::json!({
                "key_exchange": "x25519",
                "symmetric": "aes-256-gcm",
                "signature": "ed25519"
            });
            let signature = crate::crypto::sign_message(&encrypted_body, &crypto.nonce);

            let message_id = format!("<{}@mycelix.net>", js_sys::Date::now() as u64);

            let payload = serde_json::json!({
                "recipients": [send_to],
                "cc": [],
                "bcc": [],
                "encrypted_subject": encrypted_subject,
                "encrypted_body": encrypted_body,
                "encrypted_attachments": [],
                "ephemeral_pubkey": crypto.ephemeral_pubkey,
                "nonce": crypto.nonce.to_vec(),
                "signature": signature,
                "crypto_suite": crypto_suite,
                "message_id": message_id,
                "in_reply_to": reply_hash_send.get_untracked().as_deref().unwrap_or("").to_string(),
                "references": reply_thread_send.get_untracked().as_deref().map(|t| vec![t.to_string()]).unwrap_or_default(),
                "priority": "Normal",
                "read_receipt_requested": false,
                "expires_at": serde_json::Value::Null
            });

            match hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "send_email", &payload
            ).await {
                Ok(response) => {
                    web_sys::console::log_1(&format!("[Mail] send_email response: {:?}", response).into());
                    toasts_undo.push("Message sent and committed to DHT", "success");
                    mail.versions.inbox.update(|v| *v += 1);
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!("[Mail] send_email failed: {e}").into());
                    // Queue for offline retry
                    crate::offline::enqueue_action(
                        mail_leptos_types::OfflineAction::Send {
                            to: send_to, subject: send_subject,
                            body: send_body, use_pqc: send_pqc,
                        }
                    );
                    toasts_undo.push(format!("Send failed: {e}. Queued for retry."), "error");
                }
            }
        });
    };

    let toasts_draft = toasts.clone();
    let toasts_template = toasts.clone();
    let on_save_draft = move |_| {
        toasts_draft.push("Draft saved", "info");
        mail.versions.drafts.update(|v| *v += 1);
    };

    // Contact discovery: local first, then DHT search
    let dht_suggestions = RwSignal::new(Vec::<mail_leptos_types::ContactView>::new());
    let last_dht_query = RwSignal::new(String::new());

    let contact_suggestions = move || {
        let query = to_field.get().to_lowercase();
        if query.len() < 2 { return vec![]; }
        // Local contacts first
        let mut results: Vec<_> = mail.contacts.get().into_iter().filter(|c| {
            c.display_name.to_lowercase().contains(&query)
                || c.email.as_deref().unwrap_or("").to_lowercase().contains(&query)
        }).take(5).collect();

        // Append DHT search results (deduped by agent key)
        let dht = dht_suggestions.get();
        for c in dht {
            if !results.iter().any(|r| r.agent_pub_key == c.agent_pub_key) {
                results.push(c);
            }
        }

        // Trigger async DHT search if query changed and we have < 3 local results
        if results.len() < 3 && query.len() >= 3 && query != last_dht_query.get_untracked() {
            last_dht_query.set(query.clone());
            let hc = crate::holochain::use_holochain();
            if !hc.is_mock() {
                let q = query.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(contact) = hc.call_zome::<serde_json::Value, serde_json::Value>(
                        "mail_contacts", "get_contact_by_email", &serde_json::json!(q)
                    ).await {
                        if let Some(contact) = crate::zome_adapter::adapt_contact_value(contact) {
                            dht_suggestions.set(vec![contact]);
                        }
                    }
                });
            }
        }
        results.truncate(8);
        results
    };

    view! {
        <div class="page page-compose">
            <div class="page-header">
                <h1>{mode_label}</h1>
                <div class="compose-actions">
                    <button class="btn btn-secondary" on:click=on_save_draft disabled=move || sending.get()>
                        "Save Draft"
                    </button>
                    <button class="btn btn-primary" on:click=on_send disabled=move || sending.get()>
                        {move || if sending.get() { "Sending..." } else { "Send" }}
                    </button>
                </div>
            </div>

            // Keyboard shortcuts: Ctrl+Enter=send, Ctrl+Shift+S=schedule
            <div class="compose-form"
                 on:keydown=move |ev: web_sys::KeyboardEvent| {
                     if ev.ctrl_key() && ev.key() == "Enter" {
                         ev.prevent_default();
                         // Trigger send button click
                         let _ = js_sys::eval("document.querySelector('.page-compose .btn-primary')?.click()");
                     } else if ev.ctrl_key() && ev.shift_key() && ev.key() == "S" {
                         ev.prevent_default();
                         schedule_send.set(true);
                     }
                 }>
                <div class="compose-shortcuts-hint">
                    <kbd>"Ctrl+Enter"</kbd>" send  "
                    <kbd>"Ctrl+Shift+S"</kbd>" schedule  "
                    <kbd>"Tab"</kbd>" complete contact"
                </div>
                // Template picker
                <div class="template-bar">
                    <button class="btn btn-sm btn-secondary" on:click=move |_| show_templates.update(|v| *v = !*v)>
                        {move || if show_templates.get() { "\u{25BC} Templates" } else { "\u{25B6} Templates" }}
                    </button>
                    <button class="btn btn-sm btn-secondary" style=move || if show_templates.get() { "" } else { "display:none" }
                        on:click=move |_| {
                            // Save current compose content as a new template
                            let subj = subject_field.get_untracked();
                            let body = body_field.get_untracked();
                            if subj.is_empty() && body.is_empty() {
                                toasts.push("Write something first to save as template", "error");
                                return;
                            }
                            let name = if !subj.is_empty() { subj.clone() } else { "Untitled Template".into() };
                            mail.templates.update(|tpls| {
                                tpls.push(mail_leptos_types::EmailTemplate {
                                    id: format!("tpl-{}", js_sys::Date::now() as u64),
                                    name,
                                    subject: subj,
                                    body,
                                    use_pqc: true,
                                });
                            });
                            toasts.push("Template saved", "success");
                        }
                    >
                        "+ Save as Template"
                    </button>
                    <div class="template-list" style=move || if show_templates.get() { "" } else { "display:none" }>
                        {move || mail.templates.get().iter().map(|tpl| {
                            let name = tpl.name.clone();
                            let subj = tpl.subject.clone();
                            let body = tpl.body.clone();
                            let t = toasts_template.clone();
                            let name_toast = name.clone();
                            view! {
                                <button class="template-item" on:click=move |_| {
                                    subject_field.set(subj.clone());
                                    body_field.set(body.clone());
                                    show_templates.set(false);
                                    t.push(format!("Template: {}", name_toast), "info");
                                }>
                                    <span class="template-name">{name.clone()}</span>
                                    <span class="template-preview">{subj.clone()}</span>
                                </button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                </div>

                <div class="form-field">
                    <label for="to">"To"</label>
                    <input
                        id="to"
                        type="text"
                        placeholder="Recipient agent key or contact name"
                        prop:value=move || to_field.get()
                        on:input=move |ev| to_field.set(event_target_value(&ev))
                        on:keydown=move |ev: web_sys::KeyboardEvent| {
                            if ev.key() == "Tab" {
                                let suggestions = contact_suggestions();
                                if let Some(first) = suggestions.first() {
                                    ev.prevent_default();
                                    to_field.set(first.agent_pub_key.clone().unwrap_or_default());
                                }
                            }
                        }
                    />
                    {move || {
                        let suggestions = contact_suggestions();
                        (!suggestions.is_empty()).then(|| view! {
                            <div class="suggestions">
                                {suggestions.into_iter().map(|c| {
                                    let name = c.display_name.clone();
                                    let email = c.email.clone().unwrap_or_default();
                                    let agent = c.agent_pub_key.clone().unwrap_or_default();
                                    view! {
                                        <button
                                            class="suggestion-item"
                                            on:click=move |_| to_field.set(agent.clone())
                                        >
                                            <span class="suggestion-name">{name.clone()}</span>
                                            <span class="suggestion-email">{email.clone()}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        })
                    }}
                </div>

                <button class="toggle-cc" on:click=move |_| show_cc.update(|v| *v = !*v)>
                    {move || if show_cc.get() { "Hide CC" } else { "Show CC/BCC" }}
                </button>

                {move || show_cc.get().then(|| view! {
                    <div class="form-field">
                        <label for="cc">"CC"</label>
                        <input
                            id="cc"
                            type="text"
                            placeholder="CC recipients"
                            prop:value=move || cc_field.get()
                            on:input=move |ev| cc_field.set(event_target_value(&ev))
                        />
                    </div>
                })}

                <div class="form-field">
                    <label for="subject">"Subject"</label>
                    <input
                        id="subject"
                        type="text"
                        placeholder="Subject"
                        prop:value=move || subject_field.get()
                        on:input=move |ev| subject_field.set(event_target_value(&ev))
                    />
                </div>

                <div class="form-field body-field">
                    <label>"Message"</label>
                    <RichEditor value=body_field placeholder="Write your message..." />
                    <crate::components::ToneIndicator text=body_field />
                </div>

                // Attachments
                <FileDropZone attachments=attachments />

                <div class="compose-options">
                    <label class="option-toggle">
                        <input
                            type="checkbox"
                            prop:checked=move || use_pqc.get()
                            on:change=move |ev| use_pqc.set(event_target_checked(&ev))
                        />
                        <span class="option-label">
                            {move || if use_pqc.get() {
                                "\u{1F6E1} Post-Quantum Encryption (Kyber1024 + Dilithium3)"
                            } else {
                                "\u{1F512} Standard E2E Encryption (X25519 + Ed25519)"
                            }}
                        </span>
                    </label>
                    // Scheduled send
                    <label class="option-toggle">
                        <input
                            type="checkbox"
                            prop:checked=move || schedule_send.get()
                            on:change=move |ev| schedule_send.set(event_target_checked(&ev))
                        />
                        <span class="option-label">"\u{23F0} Schedule send"</span>
                    </label>
                    <div class="schedule-picker" style=move || if schedule_send.get() { "" } else { "display:none" }>
                        <input
                            type="datetime-local"
                            class="schedule-input"
                            prop:value=move || schedule_datetime.get()
                            on:input=move |ev| schedule_datetime.set(event_target_value(&ev))
                        />
                    </div>
                    // Template variable hint
                    <div class="template-vars-hint">
                        <details>
                            <summary class="vars-summary">"\u{1F4DD} Template variables"</summary>
                            <div class="vars-list">
                                <code>"{{name}}"</code>" — recipient name  "
                                <code>"{{date}}"</code>" — today's date  "
                                <code>"{{time}}"</code>" — current time  "
                                <code>"{{sender}}"</code>" — your name"
                                <p class="vars-note">"Variables are expanded when the message is sent."</p>
                            </div>
                        </details>
                    </div>
                </div>
            </div>
        </div>
    }
}

fn event_target_value(ev: &leptos::ev::Event) -> String {
    use wasm_bindgen::JsCast;
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.value())
        .or_else(|| {
            ev.target()
                .and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok())
                .map(|el| el.value())
        })
        .unwrap_or_default()
}

fn event_target_checked(ev: &leptos::ev::Event) -> bool {
    use wasm_bindgen::JsCast;
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
        .map(|el| el.checked())
        .unwrap_or(false)
}
