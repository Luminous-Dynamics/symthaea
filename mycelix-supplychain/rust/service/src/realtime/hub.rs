// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-Time Collaboration Hub
//!
//! Central manager for all WebSocket connections, channels, and document collaboration

use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use super::events::*;
use super::session::{SessionStats, WsSession};

/// Collaboration Hub - manages all real-time connections
pub struct CollaborationHub {
    /// All active sessions by session ID
    sessions: Arc<RwLock<HashMap<Uuid, WsSession>>>,

    /// Channel subscriptions: channel -> set of session IDs
    channels: Arc<RwLock<HashMap<String, HashSet<Uuid>>>>,

    /// Document editing: (doc_type, doc_id) -> set of session IDs
    documents: Arc<RwLock<HashMap<(DocumentType, Uuid), HashSet<Uuid>>>>,

    /// Document versions for optimistic locking
    document_versions: Arc<RwLock<HashMap<Uuid, i32>>>,
}

impl CollaborationHub {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            channels: Arc::new(RwLock::new(HashMap::new())),
            documents: Arc::new(RwLock::new(HashMap::new())),
            document_versions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new session
    pub async fn register_session(
        &self,
        user_id: Uuid,
        user_name: String,
        tenant_id: Uuid,
        sender: mpsc::UnboundedSender<ServerMessage>,
    ) -> Uuid {
        let session = WsSession::new(user_id, user_name, tenant_id, sender);
        let session_id = session.id;

        // Auto-subscribe to user and tenant notifications
        let user_channel = Channel::UserNotifications(user_id).to_string();
        let tenant_channel = Channel::TenantNotifications(tenant_id).to_string();
        let dashboard_channel = Channel::Dashboard(tenant_id).to_string();

        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id, session);
        }

        // Auto-subscribe to default channels
        self.subscribe(session_id, &user_channel).await;
        self.subscribe(session_id, &tenant_channel).await;
        self.subscribe(session_id, &dashboard_channel).await;

        // Send connected message
        let _ = self.send_to_session(
            session_id,
            ServerMessage::Connected(ConnectionInfo {
                session_id,
                user_id,
                tenant_id,
                connected_at: Utc::now(),
            }),
        ).await;

        session_id
    }

    /// Unregister a session (disconnect)
    pub async fn unregister_session(&self, session_id: Uuid) {
        let session = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(&session_id)
        };

        if let Some(session) = session {
            // Remove from all channels
            {
                let mut channels = self.channels.write().await;
                for channel in &session.subscriptions {
                    if let Some(subs) = channels.get_mut(channel) {
                        subs.remove(&session_id);
                    }
                }
            }

            // Remove from all documents and notify others
            for (doc_type, doc_id) in &session.active_documents {
                self.leave_document_internal(session_id, session.user_id, *doc_id, doc_type.clone()).await;
            }
        }
    }

    /// Subscribe a session to a channel
    pub async fn subscribe(&self, session_id: Uuid, channel: &str) -> bool {
        // Add to session subscriptions
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.subscribe(channel.to_string());
            } else {
                return false;
            }
        }

        // Add to channel subscribers
        {
            let mut channels = self.channels.write().await;
            channels
                .entry(channel.to_string())
                .or_insert_with(HashSet::new)
                .insert(session_id);
        }

        // Send confirmation
        let _ = self.send_to_session(
            session_id,
            ServerMessage::Subscribed(SubscriptionInfo {
                channel: channel.to_string(),
                subscribed_at: Utc::now(),
            }),
        ).await;

        true
    }

    /// Unsubscribe a session from a channel
    pub async fn unsubscribe(&self, session_id: Uuid, channel: &str) -> bool {
        // Remove from session subscriptions
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.unsubscribe(channel);
            }
        }

        // Remove from channel subscribers
        {
            let mut channels = self.channels.write().await;
            if let Some(subs) = channels.get_mut(channel) {
                subs.remove(&session_id);
            }
        }

        // Send confirmation
        let _ = self.send_to_session(
            session_id,
            ServerMessage::Unsubscribed { channel: channel.to_string() },
        ).await;

        true
    }

    /// Join a document for collaborative editing
    pub async fn join_document(
        &self,
        session_id: Uuid,
        doc_type: DocumentType,
        doc_id: Uuid,
    ) -> Result<DocumentInfo, String> {
        let (user_presence, version) = {
            let mut sessions = self.sessions.write().await;
            let session = sessions.get_mut(&session_id)
                .ok_or("Session not found")?;

            session.join_document(doc_type.clone(), doc_id);
            (session.to_presence(), 1) // Get current version from DB in production
        };

        // Add to document editors
        {
            let mut documents = self.documents.write().await;
            documents
                .entry((doc_type.clone(), doc_id))
                .or_insert_with(HashSet::new)
                .insert(session_id);
        }

        // Get all current users on this document
        let current_users = self.get_document_users(&doc_type, doc_id).await;

        // Notify other users
        self.broadcast_to_document(
            &doc_type,
            doc_id,
            ServerMessage::UserJoined(user_presence.clone()),
            Some(session_id),
        ).await;

        // Subscribe to document channel
        let doc_channel = Channel::Document {
            document_type: doc_type.clone(),
            document_id: doc_id,
        }.to_string();
        self.subscribe(session_id, &doc_channel).await;

        Ok(DocumentInfo {
            document_type: doc_type,
            document_id: doc_id,
            version,
            current_users,
        })
    }

    /// Leave a document
    pub async fn leave_document(&self, session_id: Uuid, doc_id: Uuid) -> Result<(), String> {
        let (user_id, doc_type) = {
            let mut sessions = self.sessions.write().await;
            let session = sessions.get_mut(&session_id)
                .ok_or("Session not found")?;

            let doc_type = session.active_documents.iter()
                .find(|(_, id)| id == &doc_id)
                .map(|(t, _)| t.clone())
                .ok_or("Not in document")?;

            session.leave_document(doc_id);
            (session.user_id, doc_type)
        };

        self.leave_document_internal(session_id, user_id, doc_id, doc_type).await;

        Ok(())
    }

    async fn leave_document_internal(
        &self,
        session_id: Uuid,
        user_id: Uuid,
        doc_id: Uuid,
        doc_type: DocumentType,
    ) {
        // Remove from document editors
        {
            let mut documents = self.documents.write().await;
            if let Some(editors) = documents.get_mut(&(doc_type.clone(), doc_id)) {
                editors.remove(&session_id);
            }
        }

        // Notify other users
        self.broadcast_to_document(
            &doc_type,
            doc_id,
            ServerMessage::UserLeft { user_id, document_id: doc_id },
            Some(session_id),
        ).await;

        // Unsubscribe from document channel
        let doc_channel = Channel::Document {
            document_type: doc_type,
            document_id: doc_id,
        }.to_string();
        self.unsubscribe(session_id, &doc_channel).await;

        // Send confirmation
        let _ = self.send_to_session(
            session_id,
            ServerMessage::DocumentLeft { document_id: doc_id },
        ).await;
    }

    /// Broadcast cursor update
    pub async fn broadcast_cursor(&self, session_id: Uuid, update: CursorUpdate) {
        let doc_type_opt = {
            let sessions = self.sessions.read().await;
            if let Some(session) = sessions.get(&session_id) {
                session.active_documents.iter()
                    .find(|(_, id)| id == &update.document_id)
                    .map(|(t, _)| t.clone())
            } else {
                None
            }
        };

        if let Some(doc_type) = doc_type_opt {
            self.broadcast_to_document(
                &doc_type,
                update.document_id,
                ServerMessage::UserCursor(update),
                Some(session_id),
            ).await;
        }
    }

    /// Handle field update
    pub async fn handle_field_update(
        &self,
        session_id: Uuid,
        request: FieldUpdateRequest,
    ) -> Result<FieldUpdate, String> {
        let (user_id, doc_type) = {
            let sessions = self.sessions.read().await;
            let session = sessions.get(&session_id)
                .ok_or("Session not found")?;

            let doc_type = session.active_documents.iter()
                .find(|(_, id)| id == &request.document_id)
                .map(|(t, _)| t.clone())
                .ok_or("Not editing this document")?;

            (session.user_id, doc_type)
        };

        // Check version for optimistic locking
        let new_version = {
            let mut versions = self.document_versions.write().await;
            let current = versions.entry(request.document_id).or_insert(0);
            if request.version < *current {
                return Err("Version conflict".to_string());
            }
            *current += 1;
            *current
        };

        let update = FieldUpdate {
            document_id: request.document_id,
            field_path: request.field_path,
            value: request.value,
            updated_by: user_id,
            updated_at: Utc::now(),
            version: new_version,
        };

        // Broadcast to all editors
        self.broadcast_to_document(
            &doc_type,
            request.document_id,
            ServerMessage::FieldUpdated(update.clone()),
            None, // Include sender
        ).await;

        Ok(update)
    }

    /// Broadcast a business event
    pub async fn broadcast_event(&self, tenant_id: Uuid, event: BusinessEvent) {
        let channel = Channel::TenantNotifications(tenant_id).to_string();
        self.broadcast_to_channel(&channel, ServerMessage::BusinessEvent(event)).await;
    }

    /// Get users currently editing a document
    async fn get_document_users(&self, doc_type: &DocumentType, doc_id: Uuid) -> Vec<UserPresence> {
        let sessions = self.sessions.read().await;
        let documents = self.documents.read().await;

        if let Some(editors) = documents.get(&(doc_type.clone(), doc_id)) {
            editors.iter()
                .filter_map(|sid| sessions.get(sid))
                .map(|s| s.to_presence())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Broadcast to all subscribers of a channel
    async fn broadcast_to_channel(&self, channel: &str, message: ServerMessage) {
        let channel_subs = {
            let channels = self.channels.read().await;
            channels.get(channel).cloned()
        };

        if let Some(subs) = channel_subs {
            let sessions = self.sessions.read().await;
            for session_id in subs {
                if let Some(session) = sessions.get(&session_id) {
                    let _ = session.send(message.clone());
                }
            }
        }
    }

    /// Broadcast to all editors of a document
    async fn broadcast_to_document(
        &self,
        doc_type: &DocumentType,
        doc_id: Uuid,
        message: ServerMessage,
        exclude: Option<Uuid>,
    ) {
        let editors = {
            let documents = self.documents.read().await;
            documents.get(&(doc_type.clone(), doc_id)).cloned()
        };

        if let Some(editors) = editors {
            let sessions = self.sessions.read().await;
            for session_id in editors {
                if exclude.map_or(true, |ex| ex != session_id) {
                    if let Some(session) = sessions.get(&session_id) {
                        let _ = session.send(message.clone());
                    }
                }
            }
        }
    }

    /// Send to a specific session
    async fn send_to_session(&self, session_id: Uuid, message: ServerMessage) -> Result<(), String> {
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(&session_id) {
            session.send(message).map_err(|e| e.to_string())
        } else {
            Err("Session not found".to_string())
        }
    }

    /// Get hub statistics
    pub async fn get_stats(&self) -> SessionStats {
        let sessions = self.sessions.read().await;
        let channels = self.channels.read().await;
        let documents = self.documents.read().await;

        let active_users: HashSet<Uuid> = sessions.values().map(|s| s.user_id).collect();
        let active_tenants: HashSet<Uuid> = sessions.values().map(|s| s.tenant_id).collect();
        let total_subscriptions: usize = channels.values().map(|s| s.len()).sum();

        SessionStats {
            total_connections: sessions.len(),
            active_users: active_users.len(),
            active_tenants: active_tenants.len(),
            total_subscriptions,
            active_documents: documents.len(),
        }
    }

    /// Handle client message
    pub async fn handle_message(
        &self,
        session_id: Uuid,
        message: ClientMessage,
    ) -> Result<(), String> {
        // Update activity
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.update_activity();
            }
        }

        match message {
            ClientMessage::Subscribe(req) => {
                self.subscribe(session_id, &req.channel).await;
            }
            ClientMessage::Unsubscribe(req) => {
                self.unsubscribe(session_id, &req.channel).await;
            }
            ClientMessage::JoinDocument(req) => {
                match self.join_document(session_id, req.document_type, req.document_id).await {
                    Ok(info) => {
                        let _ = self.send_to_session(session_id, ServerMessage::DocumentJoined(info)).await;
                    }
                    Err(e) => {
                        let _ = self.send_to_session(session_id, ServerMessage::Error(ErrorMessage {
                            code: "JOIN_FAILED".to_string(),
                            message: e,
                            details: None,
                        })).await;
                    }
                }
            }
            ClientMessage::LeaveDocument(req) => {
                let _ = self.leave_document(session_id, req.document_id).await;
            }
            ClientMessage::CursorMove(update) => {
                self.broadcast_cursor(session_id, update).await;
            }
            ClientMessage::FieldUpdate(req) => {
                match self.handle_field_update(session_id, req).await {
                    Ok(_) => {}
                    Err(e) => {
                        let _ = self.send_to_session(session_id, ServerMessage::Error(ErrorMessage {
                            code: "UPDATE_FAILED".to_string(),
                            message: e,
                            details: None,
                        })).await;
                    }
                }
            }
            ClientMessage::Ping => {
                let _ = self.send_to_session(session_id, ServerMessage::Pong).await;
            }
        }

        Ok(())
    }
}

impl Default for CollaborationHub {
    fn default() -> Self {
        Self::new()
    }
}
