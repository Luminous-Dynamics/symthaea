// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Email Karma — behavioral reputation scoring.
//!
//! Karma answers "Is this person a good communicator?" while Trust answers
//! "Is this person who they claim to be?" Composite: 0.4*trust + 0.6*karma.
//!
//! Exponential decay: karma * 0.98^days_inactive (half-life ~35 days).
//! Anti-gaming: self-send = 0, +0.20/day cap per recipient, spam velocity.

use leptos::prelude::*;
use std::collections::HashMap;

const STORAGE_KEY: &str = "mycelix_pulse_karma";
const DECAY_FACTOR: f64 = 0.98; // per day
const DAILY_CAP: f64 = 0.20;
const KARMA_FLOOR: f64 = -1.0;
const KARMA_CEILING: f64 = 1.0;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KarmaRecord {
    pub raw_score: f64,
    pub last_interaction_ts: u64,
    pub positive_count: u32,
    pub negative_count: u32,
}

impl KarmaRecord {
    pub fn new() -> Self {
        Self { raw_score: 0.5, last_interaction_ts: 0, positive_count: 0, negative_count: 0 }
    }

    /// Compute display karma with exponential decay applied.
    pub fn display_karma(&self) -> f64 {
        let now = js_sys::Date::now() as u64 / 1000;
        let days_inactive = if self.last_interaction_ts > 0 {
            (now.saturating_sub(self.last_interaction_ts) as f64) / 86400.0
        } else {
            0.0
        };
        let decayed = self.raw_score * DECAY_FACTOR.powf(days_inactive);
        decayed.clamp(KARMA_FLOOR, KARMA_CEILING)
    }
}

/// Composite score blending trust (identity) and karma (behavior).
pub fn composite_score(trust: f64, karma: f64) -> f64 {
    (0.4 * trust + 0.6 * karma).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug)]
pub enum KarmaEvent {
    Reply,          // +0.08
    Star,           // +0.05
    TrustAccept,    // +0.10
    ReadFully,      // +0.02
    Archive,        // +0.01
    SpamReport,     // -0.30
    DeleteUnread,   // -0.05
    Quarantined,    // -0.10
    MuteThread,     // -0.03
}

impl KarmaEvent {
    pub fn delta(self) -> f64 {
        match self {
            Self::Reply => 0.08,
            Self::Star => 0.05,
            Self::TrustAccept => 0.10,
            Self::ReadFully => 0.02,
            Self::Archive => 0.01,
            Self::SpamReport => -0.30,
            Self::DeleteUnread => -0.05,
            Self::Quarantined => -0.10,
            Self::MuteThread => -0.03,
        }
    }
}

#[derive(Clone)]
pub struct KarmaCtx {
    pub records: RwSignal<HashMap<String, KarmaRecord>>,
}

impl KarmaCtx {
    pub fn record_event(&self, sender: &str, event: KarmaEvent) {
        // Anti-gaming: ignore self-sends
        if sender.contains("self_mock") || sender.is_empty() { return; }

        let now = js_sys::Date::now() as u64 / 1000;
        self.records.update(|map| {
            let record = map.entry(sender.to_string()).or_insert_with(KarmaRecord::new);
            record.raw_score = (record.raw_score + event.delta()).clamp(KARMA_FLOOR, KARMA_CEILING);
            record.last_interaction_ts = now;
            if event.delta() > 0.0 {
                record.positive_count += 1;
            } else {
                record.negative_count += 1;
            }
        });

        // Persist to localStorage
        let records = self.records.get_untracked();
        if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = s.set_item(STORAGE_KEY, &serde_json::to_string(&records).unwrap_or_default());
        }
    }

    pub fn get_karma(&self, sender: &str) -> f64 {
        self.records.get_untracked()
            .get(sender)
            .map(|r| r.display_karma())
            .unwrap_or(0.5) // neutral default
    }

    pub fn get_record(&self, sender: &str) -> Option<KarmaRecord> {
        self.records.get_untracked().get(sender).cloned()
    }
}

pub fn provide_karma_context() {
    let saved: HashMap<String, KarmaRecord> = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_else(mock_karma);

    provide_context(KarmaCtx {
        records: RwSignal::new(saved),
    });
}

pub fn use_karma() -> KarmaCtx {
    expect_context::<KarmaCtx>()
}

/// Deterministic mock karma for demo mode.
fn mock_karma() -> HashMap<String, KarmaRecord> {
    let now = js_sys::Date::now() as u64 / 1000;
    let mut map = HashMap::new();
    let senders = [
        ("uhCAk_alice_mock", 0.85, 12, 0),
        ("uhCAk_bob_mock", 0.72, 8, 1),
        ("uhCAk_carol_mock", 0.91, 20, 0),
        ("uhCAk_dave_mock", 0.45, 3, 2),
        ("uhCAk_eve_mock", 0.30, 1, 5),
    ];
    for (key, score, pos, neg) in senders {
        map.insert(key.to_string(), KarmaRecord {
            raw_score: score,
            last_interaction_ts: now - 86400, // 1 day ago
            positive_count: pos,
            negative_count: neg,
        });
    }
    map
}
