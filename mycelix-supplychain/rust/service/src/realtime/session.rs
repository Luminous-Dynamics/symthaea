// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebSocket Session Management
//!
//! Manages individual WebSocket connections and their state

use chrono::{DateTime, Utc};
use std::collections::HashSet;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::events::{DocumentType, ServerMessage, UserPresence};

/// WebSocket session state
pub struct WsSession {
    pub id: Uuid,
    pub user_id: Uuid,
    pub user_name: String,
    pub tenant_id: Uuid,
    pub color: String,
    pub connected_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,

    /// Channels this session is subscribed to
    pub subscriptions: HashSet<String>,

    /// Documents this session is editing
    pub active_documents: HashSet<(DocumentType, Uuid)>,

    /// Sender for this session
    pub sender: mpsc::UnboundedSender<ServerMessage>,
}

impl WsSession {
    pub fn new(
        user_id: Uuid,
        user_name: String,
        tenant_id: Uuid,
        sender: mpsc::UnboundedSender<ServerMessage>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            user_name,
            tenant_id,
            color: generate_user_color(&user_id),
            connected_at: Utc::now(),
            last_active: Utc::now(),
            subscriptions: HashSet::new(),
            active_documents: HashSet::new(),
            sender,
        }
    }

    pub fn subscribe(&mut self, channel: String) -> bool {
        self.subscriptions.insert(channel)
    }

    pub fn unsubscribe(&mut self, channel: &str) -> bool {
        self.subscriptions.remove(channel)
    }

    pub fn join_document(&mut self, doc_type: DocumentType, doc_id: Uuid) -> bool {
        self.active_documents.insert((doc_type, doc_id))
    }

    pub fn leave_document(&mut self, doc_id: Uuid) -> bool {
        self.active_documents.retain(|(_, id)| id != &doc_id);
        true
    }

    pub fn is_subscribed(&self, channel: &str) -> bool {
        self.subscriptions.contains(channel)
    }

    pub fn is_editing(&self, doc_id: Uuid) -> bool {
        self.active_documents.iter().any(|(_, id)| id == &doc_id)
    }

    pub fn update_activity(&mut self) {
        self.last_active = Utc::now();
    }

    pub fn send(&self, message: ServerMessage) -> Result<(), mpsc::error::SendError<ServerMessage>> {
        self.sender.send(message)
    }

    pub fn to_presence(&self) -> UserPresence {
        UserPresence {
            user_id: self.user_id,
            user_name: self.user_name.clone(),
            color: self.color.clone(),
            joined_at: self.connected_at,
            last_active: self.last_active,
        }
    }
}

/// Generate a consistent color for a user based on their ID
fn generate_user_color(user_id: &Uuid) -> String {
    let colors = [
        "#FF6B6B", // Red
        "#4ECDC4", // Teal
        "#45B7D1", // Blue
        "#96CEB4", // Green
        "#FFEAA7", // Yellow
        "#DDA0DD", // Plum
        "#98D8C8", // Mint
        "#F7DC6F", // Gold
        "#BB8FCE", // Purple
        "#85C1E9", // Light Blue
    ];

    let bytes = user_id.as_bytes();
    let index = (bytes[0] as usize + bytes[1] as usize) % colors.len();
    colors[index].to_string()
}

/// Session statistics
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub total_connections: usize,
    pub active_users: usize,
    pub active_tenants: usize,
    pub total_subscriptions: usize,
    pub active_documents: usize,
}
