// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sensorium summary adapter for the Pulse shell.
//!
//! This keeps Pulse's local reactive mail state aligned with the cross-domain
//! `domain-mail` summary contract consumed by Mycelix Sensorium.

use std::collections::HashMap;

use domain_mail::types::{InboxSummary, InboxSummarySource, ThreadSummary, TrustHealth, TrustTier};
use leptos::prelude::*;
use mail_leptos_types::{ContactView, EmailListItem, EmailPriority};

use crate::mail_context::{use_mail, MailCtx};
use crate::offline::{use_offline, OfflineState};

pub const PULSE_SUMMARY_STORAGE_KEY: &str = "mycelix:pulse:inbox_summary:v1";

#[derive(Clone, Copy)]
pub struct PulseSummaryCtx {
    pub inbox_summary: RwSignal<InboxSummary>,
}

pub fn provide_pulse_summary_context() {
    let mail = use_mail();
    let offline = use_offline();
    let initial_summary = build_pulse_inbox_summary(
        &mail.inbox.get_untracked(),
        &mail.contacts.get_untracked(),
        &mail.sender_trust.get_untracked(),
        offline.queue_size.get_untracked(),
        InboxSummarySource::Local,
    );
    persist_pulse_summary(&initial_summary);
    let inbox_summary = RwSignal::new(initial_summary);

    Effect::new(move |_| {
        let summary = summary_from_contexts(&mail, offline);
        persist_pulse_summary(&summary);
        inbox_summary.set(summary);
    });

    provide_context(PulseSummaryCtx { inbox_summary });
}

pub fn use_pulse_summary() -> PulseSummaryCtx {
    expect_context::<PulseSummaryCtx>()
}

pub fn persist_pulse_summary(summary: &InboxSummary) {
    let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
        return;
    };
    if let Ok(json) = serde_json::to_string(summary) {
        let _ = storage.set_item(PULSE_SUMMARY_STORAGE_KEY, &json);
    }
}

pub fn summary_from_contexts(mail: &MailCtx, offline: OfflineState) -> InboxSummary {
    build_pulse_inbox_summary(
        &mail.inbox.get(),
        &mail.contacts.get(),
        &mail.sender_trust.get(),
        offline.queue_size.get(),
        InboxSummarySource::Local,
    )
}

pub fn build_pulse_inbox_summary(
    inbox: &[EmailListItem],
    contacts: &[ContactView],
    sender_trust: &HashMap<String, f64>,
    queued_actions: usize,
    source: InboxSummarySource,
) -> InboxSummary {
    let unread_count = inbox.iter().filter(|email| !email.is_read).count() as u32;
    let urgent_count = inbox
        .iter()
        .filter(|email| email.priority == EmailPriority::Urgent)
        .count() as u32;
    let high_priority_count = inbox
        .iter()
        .filter(|email| matches!(email.priority, EmailPriority::High | EmailPriority::Urgent))
        .count() as u32;

    let mut recent = inbox.to_vec();
    recent.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let recent_threads = recent
        .into_iter()
        .take(8)
        .map(|email| email_to_thread_summary(email, sender_trust))
        .collect::<Vec<_>>();

    InboxSummary {
        unread_count,
        total_count: inbox.len() as u32,
        urgent_count,
        high_priority_count,
        queued_actions: queued_actions as u32,
        recent_threads,
        trust_health: build_trust_health(inbox, contacts, sender_trust),
        source,
        updated_at: inbox
            .iter()
            .map(|email| seconds_to_micros(email.timestamp))
            .max(),
    }
}

fn email_to_thread_summary(
    email: EmailListItem,
    sender_trust: &HashMap<String, f64>,
) -> ThreadSummary {
    let trust_score = sender_trust.get(&email.sender).copied();

    ThreadSummary {
        id: email.thread_id.unwrap_or_else(|| email.hash.clone()),
        subject: email.subject.unwrap_or_else(|| "(encrypted)".to_string()),
        from_name: email.sender_name.unwrap_or_else(|| email.sender.clone()),
        from_email: email.sender,
        preview: email.snippet.unwrap_or_default(),
        trust_tier: trust_tier(trust_score),
        timestamp: email.timestamp.min(i64::MAX as u64) as i64,
        is_read: email.is_read,
    }
}

fn build_trust_health(
    inbox: &[EmailListItem],
    contacts: &[ContactView],
    sender_trust: &HashMap<String, f64>,
) -> TrustHealth {
    let trusted_contacts = contacts
        .iter()
        .filter(|contact| contact.trust_score.unwrap_or(0.0) >= 0.8)
        .count() as u32;
    let quarantined = inbox
        .iter()
        .filter(|email| sender_trust.get(&email.sender).copied().unwrap_or(0.5) < 0.5)
        .count() as u32;
    let introductions_pending = inbox
        .iter()
        .filter(|email| !sender_trust.contains_key(&email.sender))
        .count() as u32;
    let trust_scores = inbox
        .iter()
        .filter_map(|email| sender_trust.get(&email.sender).copied())
        .collect::<Vec<_>>();
    let average_trust_score = if trust_scores.is_empty() {
        0.5
    } else {
        trust_scores.iter().sum::<f64>() / trust_scores.len() as f64
    };

    TrustHealth {
        trusted_contacts,
        quarantined,
        introductions_pending,
        average_trust_score,
    }
}

fn trust_tier(score: Option<f64>) -> TrustTier {
    match score {
        Some(score) if score >= 0.8 => TrustTier::High,
        Some(score) if score >= 0.5 => TrustTier::Medium,
        Some(_) => TrustTier::Low,
        None => TrustTier::Unknown,
    }
}

fn seconds_to_micros(timestamp_secs: u64) -> i64 {
    timestamp_secs
        .min((i64::MAX / 1_000_000) as u64)
        .saturating_mul(1_000_000) as i64
}
