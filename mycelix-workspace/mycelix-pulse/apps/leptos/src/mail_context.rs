// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mail context — provides reactive mail data via Leptos signals.
//!
//! Each data domain loads independently via its own `spawn_local` (no waterfall).
//! Mock data renders immediately; conductor data replaces it asynchronously.
//! Includes optimistic actions (star, read, archive, delete) and real-time signals.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use mail_leptos_types::*;
use std::collections::HashMap;

use std::collections::HashSet;

/// Returns true if running on a demo domain (luminousdynamics.io) — mock data only.
/// Returns false on production (mycelix.net) or localhost — real conductor required.
pub fn is_demo_mode() -> bool {
    web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .map(|h| h.contains("luminousdynamics.io"))
        .unwrap_or(false)
}
use crate::holochain::use_holochain;
use crate::mock_data;

#[derive(Clone, Copy)]
pub struct MailVersions {
    pub inbox: RwSignal<u32>,
    pub contacts: RwSignal<u32>,
    pub folders: RwSignal<u32>,
    pub drafts: RwSignal<u32>,
}

#[derive(Clone)]
pub struct MailCtx {
    pub versions: MailVersions,
    pub folders: RwSignal<Vec<FolderView>>,
    pub inbox: RwSignal<Vec<EmailListItem>>,
    pub sent: RwSignal<Vec<EmailListItem>>,
    pub contacts: RwSignal<Vec<ContactView>>,
    pub drafts: RwSignal<Vec<DraftView>>,
    pub key_status: RwSignal<BundleStatus>,
    pub active_folder: RwSignal<String>,
    pub selected_email: RwSignal<Option<String>>,
    pub compose_mode: RwSignal<ComposeMode>,
    pub search_query: RwSignal<String>,
    pub loading: RwSignal<bool>,
    /// Sender agent key → trust score (from contacts + trust zome)
    pub sender_trust: RwSignal<HashMap<String, f64>>,
    /// Typing indicators: sender → thread_id
    pub typing_indicators: RwSignal<HashMap<String, Option<String>>>,
    /// Batch selection (#3)
    pub selected_hashes: RwSignal<HashSet<String>>,
    /// Labels (#2)
    pub labels: RwSignal<Vec<LabelView>>,
    /// Filter rules (#1)
    pub filters: RwSignal<Vec<FilterRule>>,
    /// Signatures (#6)
    pub signatures: RwSignal<Vec<SignatureView>>,
    /// Vacation responder (#7)
    pub vacation: RwSignal<VacationResponder>,
    /// Reading pane position (#5)
    pub reading_pane: RwSignal<ReadingPanePosition>,
    /// Swipe actions (#10)
    pub swipe_left: RwSignal<SwipeAction>,
    pub swipe_right: RwSignal<SwipeAction>,
    /// Email templates (#6)
    pub templates: RwSignal<Vec<EmailTemplate>>,
}

impl MailCtx {
    pub fn unread_count(&self) -> usize {
        self.inbox.get_untracked().iter().filter(|e| !e.is_read).count()
    }

    /// Get emails grouped by thread. Non-threaded emails get their own group.
    pub fn threaded_emails(&self) -> Vec<ThreadView> {
        let emails = self.inbox.get();
        let mut threads: HashMap<String, Vec<EmailListItem>> = HashMap::new();
        let mut standalone = Vec::new();

        for email in emails {
            if let Some(tid) = &email.thread_id {
                threads.entry(tid.clone()).or_default().push(email);
            } else {
                standalone.push(email);
            }
        }

        let mut result: Vec<ThreadView> = threads.into_iter().map(|(tid, mut msgs)| {
            msgs.sort_by_key(|m| m.timestamp);
            let subject = msgs.first()
                .and_then(|m| m.subject.clone())
                .unwrap_or_else(|| "(encrypted)".to_string())
                // Strip "Re: " prefix for thread display
                .trim_start_matches("Re: ").to_string();
            let participants: Vec<String> = msgs.iter()
                .filter_map(|m| m.sender_name.clone())
                .collect::<Vec<_>>();
            // Deduplicate
            let mut unique_participants = Vec::new();
            for p in &participants {
                if !unique_participants.contains(p) {
                    unique_participants.push(p.clone());
                }
            }
            let last_activity = msgs.iter().map(|m| m.timestamp).max().unwrap_or(0);
            let unread_count = msgs.iter().filter(|m| !m.is_read).count() as u32;

            ThreadView {
                thread_id: tid,
                subject,
                participants: unique_participants,
                messages: msgs,
                last_activity,
                unread_count,
            }
        }).collect();

        // Add standalone emails as single-message threads
        for email in standalone {
            let subject = email.subject.clone().unwrap_or_else(|| "(encrypted)".to_string());
            let participants = email.sender_name.clone().into_iter().collect();
            let unread = if email.is_read { 0 } else { 1 };
            let ts = email.timestamp;
            result.push(ThreadView {
                thread_id: email.hash.clone(),
                subject,
                participants,
                messages: vec![email],
                last_activity: ts,
                unread_count: unread,
            });
        }

        // Sort by most recent activity
        result.sort_by(|a, b| b.last_activity.cmp(&a.last_activity));
        result
    }

    /// Get trust score for a sender agent key.
    pub fn trust_for_sender(&self, sender: &str) -> Option<f64> {
        self.sender_trust.get_untracked().get(sender).copied()
    }

    // ── Actions (optimistic UI) ──

    pub fn toggle_star(&self, hash: &str) {
        let hash = hash.to_string();
        // Fire karma event
        if let Some(sender) = self.inbox.get_untracked().iter().find(|e| e.hash == hash).map(|e| e.sender.clone()) {
            if let Ok(karma) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::karma::use_karma())) {
                karma.record_event(&sender, crate::karma::KarmaEvent::Star);
            }
        }
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.is_starred = !e.is_starred;
            }
        });
        // Fire-and-forget zome call
        let hc = use_holochain();
        let h = hash.clone();
        spawn_local(async move {
            if hc.is_mock() { return; }
            let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "update_email_state",
                &serde_json::json!([h, { "is_starred": true }]),
            ).await;
        });
    }

    pub fn toggle_read(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.is_read = !e.is_read;
            }
        });
        let hc = use_holochain();
        let h = hash.clone();
        spawn_local(async move {
            if hc.is_mock() { return; }
            let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "mark_as_read",
                &serde_json::json!([h, false]),
            ).await;
        });
    }

    pub fn archive_email(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            emails.retain(|e| e.hash != hash);
        });
        let hc = use_holochain();
        let h = hash.clone();
        spawn_local(async move {
            if hc.is_mock() { return; }
            let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "update_email_state",
                &serde_json::json!([h, { "is_archived": true }]),
            ).await;
        });
    }

    pub fn delete_email(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            emails.retain(|e| e.hash != hash);
        });
        let hc = use_holochain();
        let h = hash.clone();
        spawn_local(async move {
            if hc.is_mock() { return; }
            let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "update_email_state",
                &serde_json::json!([h, { "is_trashed": true }]),
            ).await;
        });
    }

    /// Pin/unpin a conversation.
    pub fn toggle_pin(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.is_pinned = !e.is_pinned;
            }
        });
    }

    /// Snooze email until a given timestamp (#3 from competitive gap).
    pub fn snooze_email(&self, hash: &str, until: u64) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.is_snoozed = true;
                e.snooze_until = Some(until);
            }
        });
    }

    /// Mute a conversation (#12).
    pub fn mute_thread(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            // Find the thread_id of this email
            let tid = emails.iter().find(|e| e.hash == hash).and_then(|e| e.thread_id.clone());
            if let Some(tid) = tid {
                for e in emails.iter_mut() {
                    if e.thread_id.as_deref() == Some(&tid) {
                        e.is_muted = true;
                    }
                }
            } else if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.is_muted = true;
            }
        });
    }

    /// Cycle star type (#11).
    pub fn cycle_star(&self, hash: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                if e.is_starred {
                    e.star_type = Some(e.star_type.unwrap_or_default().next());
                } else {
                    e.is_starred = true;
                    e.star_type = Some(StarType::Yellow);
                }
            }
        });
    }

    /// Apply a label to an email (#2).
    pub fn apply_label(&self, hash: &str, label: &str) {
        let hash = hash.to_string();
        let label = label.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                if !e.labels.contains(&label) {
                    e.labels.push(label);
                }
            }
        });
    }

    /// Remove a label from an email.
    pub fn remove_label(&self, hash: &str, label: &str) {
        let hash = hash.to_string();
        self.inbox.update(|emails| {
            if let Some(e) = emails.iter_mut().find(|e| e.hash == hash) {
                e.labels.retain(|l| l != label);
            }
        });
    }

    /// Get the default signature for new/reply emails.
    pub fn default_signature(&self, is_reply: bool) -> Option<String> {
        self.signatures.get_untracked().iter().find(|s| {
            if is_reply { s.use_for_reply } else { s.use_for_new }
        }).map(|s| s.body_html.clone())
    }

    /// Move email to a different folder (#7).
    pub fn move_to_folder(&self, hash: &str, folder_hash: &str) {
        let hash = hash.to_string();
        let folder = folder_hash.to_string();
        self.inbox.update(|emails| {
            emails.retain(|e| e.hash != hash);
        });
        let hc = use_holochain();
        spawn_local(async move {
            if hc.is_mock() { return; }
            let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages", "move_to_folder",
                &serde_json::json!([hash, folder]),
            ).await;
        });
    }

    // ── Batch actions (#3) ──

    pub fn toggle_selection(&self, hash: &str) {
        let hash = hash.to_string();
        self.selected_hashes.update(|set| {
            if !set.remove(&hash) { set.insert(hash); }
        });
    }

    pub fn select_all(&self) {
        let hashes: HashSet<String> = self.inbox.get_untracked().iter().map(|e| e.hash.clone()).collect();
        self.selected_hashes.set(hashes);
    }

    pub fn deselect_all(&self) {
        self.selected_hashes.set(HashSet::new());
    }

    pub fn batch_archive(&self) {
        let selected = self.selected_hashes.get_untracked();
        for hash in &selected {
            self.archive_email(hash);
        }
        self.selected_hashes.set(HashSet::new());
    }

    pub fn batch_mark_read(&self) {
        let selected = self.selected_hashes.get_untracked();
        self.inbox.update(|emails| {
            for e in emails.iter_mut() {
                if selected.contains(&e.hash) {
                    e.is_read = true;
                }
            }
        });
        self.selected_hashes.set(HashSet::new());
    }

    pub fn batch_star(&self) {
        let selected = self.selected_hashes.get_untracked();
        self.inbox.update(|emails| {
            for e in emails.iter_mut() {
                if selected.contains(&e.hash) {
                    e.is_starred = true;
                }
            }
        });
        self.selected_hashes.set(HashSet::new());
    }

    pub fn batch_delete(&self) {
        let selected = self.selected_hashes.get_untracked();
        for hash in &selected {
            self.delete_email(hash);
        }
        self.selected_hashes.set(HashSet::new());
    }

    /// Handle incoming real-time signal from conductor.
    pub fn handle_signal(&self, signal: MailSignalView) {
        match signal {
            MailSignalView::EmailReceived { email_hash, sender, sender_name, subject, timestamp, priority } => {
                // Desktop notification + sound
                crate::notifications::show_desktop_notification(
                    sender_name.as_deref().unwrap_or(&sender),
                    subject.as_deref().unwrap_or("New message"),
                );
                crate::notifications::play_notification_sound();

                self.inbox.update(|emails| {
                    emails.insert(0, EmailListItem {
                        hash: email_hash,
                        sender: sender.clone(),
                        sender_name,
                        encrypted_subject: vec![],
                        subject,
                        snippet: None,
                        timestamp,
                        priority,
                        is_read: false,
                        is_starred: false,
                        star_type: None,
                        is_pinned: false,
                        is_muted: false,
                        is_snoozed: false,
                        snooze_until: None,
                        has_attachments: false,
                        labels: vec![],
                        thread_id: None,
                        crypto_suite: CryptoSuiteView {
                            key_exchange: "x25519".into(),
                            symmetric: "chacha20-poly1305".into(),
                            signature: "ed25519".into(),
                        },
                    });
                });
            }
            MailSignalView::EmailStateChanged { email_hash, is_read, is_starred, is_archived, is_trashed } => {
                if is_archived == Some(true) || is_trashed == Some(true) {
                    self.inbox.update(|emails| emails.retain(|e| e.hash != email_hash));
                } else {
                    self.inbox.update(|emails| {
                        if let Some(e) = emails.iter_mut().find(|e| e.hash == email_hash) {
                            if let Some(r) = is_read { e.is_read = r; }
                            if let Some(s) = is_starred { e.is_starred = s; }
                        }
                    });
                }
            }
            MailSignalView::TypingIndicator { sender, thread_id } => {
                self.typing_indicators.update(|m| { m.insert(sender.clone(), thread_id); });
                // Auto-clear after 3 seconds
                let indicators = self.typing_indicators;
                spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_secs(3)).await;
                    indicators.update(|m| { m.remove(&sender); });
                });
            }
            MailSignalView::ReadReceiptReceived { .. } => {
                // Could show a "read" checkmark — skip for now
            }
        }
    }
}

pub fn provide_mail_context() {
    let versions = MailVersions {
        inbox: RwSignal::new(0),
        contacts: RwSignal::new(0),
        folders: RwSignal::new(0),
        drafts: RwSignal::new(0),
    };

    let demo = is_demo_mode();
    let ctx = MailCtx {
        versions,
        folders: RwSignal::new(if demo { mock_data::mock_folders() } else { vec![] }),
        inbox: RwSignal::new(if demo {
            mock_data::mock_inbox()
        } else {
            // Production: don't load cached mock data from localStorage
            vec![]
        }),
        sent: RwSignal::new(vec![]),
        contacts: RwSignal::new(if demo { mock_data::mock_contacts() } else { vec![] }),
        drafts: RwSignal::new(if demo { mock_data::mock_drafts() } else { vec![] }),
        key_status: RwSignal::new(BundleStatus::NoBundle),
        active_folder: RwSignal::new("Inbox".to_string()),
        selected_email: RwSignal::new(None),
        compose_mode: RwSignal::new(ComposeMode::New),
        search_query: RwSignal::new(String::new()),
        loading: RwSignal::new(true),
        sender_trust: RwSignal::new(if demo { mock_data::mock_sender_trust() } else { HashMap::new() }),
        typing_indicators: RwSignal::new(HashMap::new()),
        selected_hashes: RwSignal::new(HashSet::new()),
        labels: RwSignal::new(mock_data::mock_labels()),
        filters: RwSignal::new(mock_data::mock_filters()),
        signatures: RwSignal::new(mock_data::mock_signatures()),
        vacation: RwSignal::new(mock_data::mock_vacation()),
        reading_pane: RwSignal::new(ReadingPanePosition::Off),
        swipe_left: RwSignal::new(SwipeAction::Archive),
        swipe_right: RwSignal::new(SwipeAction::Delete),
        templates: RwSignal::new(mock_data::mock_templates()),
    };

    provide_context(ctx.clone());

    // Auto-save inbox to localStorage (demo mode only — production uses DHT)
    if demo {
        let ctx_persist = ctx.clone();
        Effect::new(move |_| {
            let emails = ctx_persist.inbox.get();
            crate::persistence::save_inbox(&emails);
        });
    }

    // Capture HolochainCtx INSIDE the reactive context (before spawn_local)
    let hc = use_holochain();

    // Wait for conductor connection to settle, then load data.
    // The BrowserWsTransport.connect() runs async in HolochainProvider.
    // We poll the status until it's no longer Connecting (max 10s).
    let ctx_load = ctx.clone();
    let hc_load = hc.clone();
    spawn_local(async move {
        // Demo mode: load mock data immediately, don't wait for conductor
        if is_demo_mode() {
            web_sys::console::log_1(&"[Mail] Demo mode — loading sample data immediately".into());
            seed_demo_mail_data(&ctx_load);
            ctx_load.loading.set(false);
            return;
        }

        // Production: wait for connection to settle with exponential backoff
        let mut delay_ms = 500u64;
        for attempt in 0..8 {
            let status = hc_load.status.get_untracked();
            if status != crate::holochain::ConnectionStatus::Connecting {
                break;
            }
            web_sys::console::log_1(&format!("[Mail] Waiting for conductor (attempt {}, {}ms)...", attempt + 1, delay_ms).into());
            gloo_timers::future::sleep(std::time::Duration::from_millis(delay_ms)).await;
            delay_ms = (delay_ms * 2).min(5000); // exponential backoff, max 5s
        }

        let status = hc_load.status.get_untracked();
        web_sys::console::log_1(&format!("[Mail] Connection settled: {:?}", status).into());

        if status == crate::holochain::ConnectionStatus::Connected {
            // Production (mycelix.net / localhost) — load from conductor
            web_sys::console::log_1(&"[Mail] Production mode — loading from conductor...".into());
            load_live_mail_data(ctx_load.clone(), &hc_load).await;
            ctx_load.loading.set(false);
        } else {
            // Production but no conductor — show empty state, retry later
            web_sys::console::log_1(&"[Mail] Conductor not available — waiting...".into());
            gloo_timers::future::sleep(std::time::Duration::from_secs(10)).await;
            if hc_load.status.get_untracked() == crate::holochain::ConnectionStatus::Connected {
                web_sys::console::log_1(&"[Mail] Late conductor connection — loading data...".into());
                load_live_mail_data(ctx_load.clone(), &hc_load).await;
            }
            ctx_load.loading.set(false);
        }
    });

    // Reactive: watch connection status — reload data if conductor connects later
    let ctx_reactive = ctx.clone();
    let hc_reactive = hc.clone();
    Effect::new(move |prev_status: Option<crate::holochain::ConnectionStatus>| {
        let current = hc_reactive.status.get();
        if prev_status == Some(crate::holochain::ConnectionStatus::Mock)
            && current == crate::holochain::ConnectionStatus::Connected
        {
            web_sys::console::log_1(&"[Mail] Conductor connected! Reloading data...".into());
            let ctx = ctx_reactive.clone();
            let hc = hc_reactive.clone();
            spawn_local(async move {
                load_live_mail_data(ctx.clone(), &hc).await;
                crate::offline::flush_queue();
                ctx.loading.set(false);
            });
        }
        current
    });

    // Reactive refresh hooks for explicit UI-triggered version bumps.
    let ctx_versions = ctx.clone();
    let hc_versions = hc.clone();
    Effect::new(move |prev: Option<(u32, u32, u32)>| {
        let current = (
            ctx_versions.versions.inbox.get(),
            ctx_versions.versions.contacts.get(),
            ctx_versions.versions.folders.get(),
        );

        if prev.is_some() && hc_versions.status.get_untracked() == crate::holochain::ConnectionStatus::Connected {
            let ctx = ctx_versions.clone();
            let hc = hc_versions.clone();
            spawn_local(async move {
                load_live_mail_data(ctx.clone(), &hc).await;
            });
        }

        current
    });

    // Wire real-time signal handler (always set up — will activate when connected)
    let ctx_signals = ctx.clone();
    hc.set_signal_handler(move |bytes: Vec<u8>| {
        // Try MessagePack first (conductor uses msgpack), then JSON fallback
        let signal = rmp_serde::from_slice::<MailSignalView>(&bytes)
            .or_else(|_| serde_json::from_slice::<MailSignalView>(&bytes));
        match signal {
            Ok(sig) => {
                web_sys::console::log_1(&format!("[Mail] Signal received: {:?}", std::mem::discriminant(&sig)).into());
                ctx_signals.handle_signal(sig);
            }
            Err(_) => {
                web_sys::console::log_1(&format!("[Mail] Unrecognized signal ({} bytes)", bytes.len()).into());
            }
        }
    });
}

async fn load_inbox(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    if hc.is_mock() {
        web_sys::console::log_1(&"[Mail] Running in mock mode — using local data".into());
        return;
    }

    web_sys::console::log_1(&"[Mail] Fetching inbox from conductor...".into());

    // Try structured response first, fall back to raw JSON logging
    match hc.call_zome::<serde_json::Value, serde_json::Value>(
        "mail_messages", "get_inbox", &serde_json::json!({ "limit": 50 })
    ).await {
        Ok(response) => {
            web_sys::console::log_1(&format!("[Mail] get_inbox raw response: {}",
                serde_json::to_string_pretty(&response).unwrap_or_else(|_| format!("{response:?}"))
            ).into());

            // Try adapter wire types first, then direct parse, then raw JSON
            let wire_data = if response.is_array() {
                response.clone()
            } else {
                response.get("records").or(response.get("items")).or(response.get("data"))
                    .cloned().unwrap_or(serde_json::Value::Null)
            };

            // Attempt 1: Parse as wire types and adapt
            if let Ok(wire_emails) = serde_json::from_value::<Vec<crate::zome_adapter::WireEmailListItem>>(wire_data.clone()) {
                let contacts = ctx.contacts.get_untracked();
                let emails = crate::zome_adapter::adapt_inbox(wire_emails, &contacts);
                if !emails.is_empty() {
                    web_sys::console::log_1(&format!("[Mail] Loaded {} emails via adapter", emails.len()).into());
                    ctx.inbox.set(emails);
                    hydrate_inbox_previews(ctx.clone(), hc.clone());
                } else {
                    web_sys::console::log_1(&"[Mail] Conductor returned empty inbox".into());
                }
            }
            // Attempt 2: Direct parse (if zome returns frontend-compatible types)
            else if let Ok(emails) = serde_json::from_value::<Vec<EmailListItem>>(wire_data) {
                if !emails.is_empty() {
                    web_sys::console::log_1(&format!("[Mail] Loaded {} emails (direct)", emails.len()).into());
                    ctx.inbox.set(emails);
                    hydrate_inbox_previews(ctx.clone(), hc.clone());
                }
            } else {
                web_sys::console::warn_1(&"[Mail] Could not parse inbox response. Keeping mock data.".into());
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Mail] get_inbox zome call failed: {e}. Keeping mock data.").into());
        }
    }
}

async fn load_contacts(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    if hc.is_mock() { return; }

    match hc.call_zome::<(), Vec<serde_json::Value>>(
        "mail_contacts", "get_all_contacts", &()
    ).await {
        Ok(records) => {
            let contacts = crate::zome_adapter::adapt_contact_values(records);

            // Build sender trust map from contacts
            let trust_map: HashMap<String, f64> = contacts.iter()
                .filter_map(|c| {
                    let key = c.agent_pub_key.as_ref()?;
                    let score = c.trust_score?;
                    Some((key.clone(), score))
                })
                .collect();

            ctx.sender_trust.set(trust_map);
            ctx.contacts.set(contacts);
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Mail] get_all_contacts failed: {e}").into());
        }
    }
}

async fn load_folders(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    if hc.is_mock() { return; }

    web_sys::console::log_1(&"[Mail] Fetching folders from conductor...".into());

    match hc.call_zome::<(), serde_json::Value>(
        "mail_messages", "get_folders", &()
    ).await {
        Ok(response) => {
            web_sys::console::log_1(&format!("[Mail] get_folders raw: {}",
                serde_json::to_string_pretty(&response).unwrap_or_else(|_| format!("{response:?}"))
            ).into());

            let folders_result = if response.is_array() {
                serde_json::from_value::<Vec<FolderView>>(response.clone())
            } else if let Some(arr) = response.get("folders").or(response.get("data")) {
                serde_json::from_value::<Vec<FolderView>>(arr.clone())
            } else {
                Err(serde_json::from_value::<Vec<FolderView>>(serde_json::Value::Null).unwrap_err())
            };

            match folders_result {
                Ok(folders) if !folders.is_empty() => {
                    web_sys::console::log_1(&format!("[Mail] Loaded {} folders from conductor", folders.len()).into());
                    ctx.folders.set(folders);
                }
                Ok(_) => web_sys::console::log_1(&"[Mail] Conductor returned empty folders".into()),
                Err(e) => web_sys::console::warn_1(&format!("[Mail] Could not parse folders: {e}. Keeping mock data.").into()),
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Mail] get_folders failed: {e}. Keeping mock data.").into());
        }
    }
}

async fn load_key_status(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    if hc.is_mock() { return; }

    match hc.call_zome::<(), serde_json::Value>(
        "mail_keys", "needs_refresh", &()
    ).await {
        Ok(record) => {
            if let Ok(status) = serde_json::from_value::<BundleStatus>(record) {
                ctx.key_status.set(status);
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Mail] needs_refresh failed: {e}").into());
        }
    }
}

/// Initialize CRDT sync state on the conductor.
async fn init_sync(hc: &crate::holochain::HolochainCtx) {
    // Initialize sync state if not already done
    match hc.call_zome::<(), serde_json::Value>("mail_sync", "get_sync_state", &()).await {
        Ok(val) if val.is_null() => {
            web_sys::console::log_1(&"[Mail] Initializing sync state...".into());
            let _ = hc.call_zome::<(), serde_json::Value>("mail_sync", "init_sync_state", &()).await;
            let _ = hc.call_zome::<bool, ()>("mail_sync", "set_online_status", &true).await;
        }
        Ok(_) => {
            // Sync state exists, mark online
            let _ = hc.call_zome::<bool, ()>("mail_sync", "set_online_status", &true).await;
            // Process any queued offline operations
            match hc.call_zome::<(), serde_json::Value>("mail_sync", "process_offline_queue", &()).await {
                Ok(val) => {
                    if let Some(n) = val.as_u64() {
                        if n > 0 {
                            web_sys::console::log_1(&format!("[Mail] Processed {n} offline operations").into());
                        }
                    }
                }
                Err(_) => {}
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Mail] sync init skipped: {e}").into());
        }
    }
}

/// Load trust scores for known agents.
async fn load_trust_scores(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    if hc.is_mock() { return; }

    // Get trust scores for each contact's agent key
    let contacts = ctx.contacts.get_untracked();
    let mut trust_map = HashMap::new();

    for contact in &contacts {
        if let Some(agent_key) = &contact.agent_pub_key {
            match hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_trust", "get_trust_score", &serde_json::json!(agent_key)
            ).await {
                Ok(score_val) => {
                    if let Some(score) = score_val.get("score").and_then(|s| s.as_f64()) {
                        trust_map.insert(agent_key.clone(), score);
                    }
                }
                Err(_) => {} // Trust score not available — that's fine
            }
        }
    }

    if !trust_map.is_empty() {
        web_sys::console::log_1(&format!("[Mail] Loaded {} trust scores", trust_map.len()).into());
        ctx.sender_trust.set(trust_map);
    }
}

fn seed_demo_mail_data(ctx: &MailCtx) {
    ctx.inbox.set(mock_data::mock_inbox());
    ctx.contacts.set(mock_data::mock_contacts());
    ctx.drafts.set(mock_data::mock_drafts());
    ctx.sender_trust.set(mock_data::mock_sender_trust());
}

async fn load_live_mail_data(ctx: MailCtx, hc: &crate::holochain::HolochainCtx) {
    init_sync(hc).await;
    load_folders(ctx.clone(), hc).await;
    load_contacts(ctx.clone(), hc).await;
    load_inbox(ctx.clone(), hc).await;
    load_key_status(ctx.clone(), hc).await;
    load_trust_scores(ctx, hc).await;
}

fn hydrate_inbox_previews(ctx: MailCtx, hc: crate::holochain::HolochainCtx) {
    spawn_local(async move {
        if hc.is_mock() {
            return;
        }

        let hashes: Vec<String> = ctx
            .inbox
            .get_untracked()
            .iter()
            .map(|e| e.hash.clone())
            .collect();

        for hash in hashes {
            let response = match hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_messages",
                "get_email",
                &serde_json::json!(hash.clone()),
            ).await {
                Ok(value) if !value.is_null() => value,
                _ => continue,
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

            let mut subject = crate::crypto::decode_transport_text(&encrypted_subject);
            let mut body = crate::crypto::decode_transport_text(&encrypted_body);

            if (subject.is_none() || body.is_none()) && ephemeral_pubkey.len() == 32 && nonce.len() == 24 {
                if let Ok(crypto) = crate::crypto::derive_message_crypto_for_local_recipient(
                    &ephemeral_pubkey,
                    &nonce,
                ) {
                    if subject.is_none() {
                        if let Ok(bytes) = crate::crypto::decrypt_with_key(
                            &encrypted_subject,
                            &crypto.subject_key,
                            &crypto.nonce,
                        ).await {
                            subject = String::from_utf8(bytes).ok();
                        }
                    }

                    if body.is_none() {
                        if let Ok(bytes) = crate::crypto::decrypt_with_key(
                            &encrypted_body,
                            &crypto.body_key,
                            &crypto.nonce,
                        ).await {
                            body = String::from_utf8(bytes).ok();
                        }
                    }
                }
            }

            let snippet = body.map(|body| {
                let clean = body.replace('\n', " ");
                if clean.len() > 140 {
                    format!("{}...", &clean[..137])
                } else {
                    clean
                }
            });

            ctx.inbox.update(|emails| {
                if let Some(email) = emails.iter_mut().find(|e| e.hash == hash) {
                    if let Some(subject) = subject.clone() {
                        email.subject = Some(subject);
                    }
                    if let Some(snippet) = snippet.clone() {
                        email.snippet = Some(snippet);
                    }
                }
            });
        }
    });
}

pub fn use_mail() -> MailCtx {
    use_context::<MailCtx>().unwrap_or_else(|| {
        // Fallback: create an empty MailCtx so components don't panic
        // before the provider has mounted.
        let versions = MailVersions {
            inbox: RwSignal::new(0),
            contacts: RwSignal::new(0),
            folders: RwSignal::new(0),
            drafts: RwSignal::new(0),
        };
        MailCtx {
            versions,
            folders: RwSignal::new(vec![]),
            inbox: RwSignal::new(vec![]),
            sent: RwSignal::new(vec![]),
            contacts: RwSignal::new(vec![]),
            drafts: RwSignal::new(vec![]),
            key_status: RwSignal::new(BundleStatus::NoBundle),
            active_folder: RwSignal::new("Inbox".to_string()),
            selected_email: RwSignal::new(None),
            compose_mode: RwSignal::new(ComposeMode::New),
            search_query: RwSignal::new(String::new()),
            loading: RwSignal::new(true),
            sender_trust: RwSignal::new(HashMap::new()),
            typing_indicators: RwSignal::new(HashMap::new()),
            selected_hashes: RwSignal::new(HashSet::new()),
            labels: RwSignal::new(vec![]),
            filters: RwSignal::new(vec![]),
            signatures: RwSignal::new(vec![]),
            vacation: RwSignal::new(VacationResponder {
                enabled: false,
                subject: String::new(),
                body: String::new(),
                start_date: None,
                end_date: None,
                contacts_only: false,
            }),
            reading_pane: RwSignal::new(ReadingPanePosition::Off),
            swipe_left: RwSignal::new(SwipeAction::Archive),
            swipe_right: RwSignal::new(SwipeAction::Delete),
            templates: RwSignal::new(vec![]),
        }
    })
}
