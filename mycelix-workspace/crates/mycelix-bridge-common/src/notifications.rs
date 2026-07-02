// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared notification helpers for bridge coordinator zomes.
//!
//! Each bridge coordinator can expose notification extern functions
//! by delegating to these helpers. The helpers handle storage,
//! indexing, and signal emission.

use serde::{Deserialize, Serialize};

/// Input for subscribing to events from specific clusters/types.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SubscribeInput {
    /// Event types to subscribe to (e.g., "property_transfer_initiated").
    /// Empty = subscribe to all.
    pub event_types: Vec<String>,
    /// Clusters to subscribe to (e.g., "commons", "civic").
    /// Empty = subscribe to all.
    pub clusters: Vec<String>,
}

/// Input for paginated notification queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationQueryInput {
    /// Maximum number to return.
    pub limit: Option<usize>,
    /// Only return unread notifications.
    pub unread_only: bool,
    /// Filter by priority (0-3). None = all priorities.
    pub min_priority: Option<u8>,
}

/// Signal emitted to connected UI clients when a notification arrives.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NotificationSignal {
    pub signal_type: String,
    pub source_cluster: String,
    pub event_type: String,
    pub payload: String,
    pub priority: u8,
}

/// Maximum notifications per agent before oldest are pruned.
pub const MAX_NOTIFICATIONS_PER_AGENT: usize = 500;

/// Maximum subscriptions per agent.
pub const MAX_SUBSCRIPTIONS_PER_AGENT: usize = 50;

/// Default notification query limit.
pub const DEFAULT_NOTIFICATION_LIMIT: usize = 50;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_input_serde_roundtrip() {
        let input = SubscribeInput {
            event_types: vec!["disaster_declared".into()],
            clusters: vec!["civic".into()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: SubscribeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_types, input.event_types);
        assert_eq!(decoded.clusters, input.clusters);
    }

    #[test]
    fn notification_query_defaults() {
        let input = NotificationQueryInput {
            limit: None,
            unread_only: false,
            min_priority: None,
        };
        assert_eq!(input.limit.unwrap_or(DEFAULT_NOTIFICATION_LIMIT), 50);
    }

    #[test]
    fn notification_signal_serde_roundtrip() {
        let signal = NotificationSignal {
            signal_type: "notification".into(),
            source_cluster: "commons".into(),
            event_type: "property_transfer".into(),
            payload: r#"{"hash":"abc"}"#.into(),
            priority: 2,
        };
        let json = serde_json::to_string(&signal).unwrap();
        let decoded: NotificationSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.source_cluster, "commons");
        assert_eq!(decoded.priority, 2);
    }
}
