// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mail types — inbox summary consumed from Mail REST API.
//! These are NOT Holochain zome types — they match the Mail backend's JSON API.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InboxSummary {
    pub unread_count: u32,
    pub total_count: u32,
    pub recent_threads: Vec<ThreadSummary>,
    pub trust_health: TrustHealth,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub id: String,
    pub subject: String,
    pub from_name: String,
    pub from_email: String,
    pub preview: String,
    pub trust_tier: TrustTier,
    pub timestamp: i64,
    pub is_read: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TrustTier { High, Medium, Low, Unknown }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustHealth {
    pub trusted_contacts: u32,
    pub quarantined: u32,
    pub introductions_pending: u32,
    pub average_trust_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComposeInput {
    pub to: String,
    pub subject: String,
    pub body: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inbox_summary_roundtrip() {
        let inbox = InboxSummary { unread_count: 5, total_count: 100, recent_threads: vec![], trust_health: TrustHealth { trusted_contacts: 50, quarantined: 2, introductions_pending: 1, average_trust_score: 0.75 } };
        let bytes = rmp_serde::to_vec_named(&inbox).unwrap();
        let decoded: InboxSummary = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.unread_count, 5);
        assert_eq!(decoded.trust_health.trusted_contacts, 50);
    }
}
