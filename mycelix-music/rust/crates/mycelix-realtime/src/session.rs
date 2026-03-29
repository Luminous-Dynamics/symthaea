// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Session management for real-time collaboration

use crate::{RealtimeError, RealtimeResult, SessionType, Participant};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Created,
    Active,
    Paused,
    Ending,
    Ended,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub session_type: SessionType,
    pub max_participants: usize,
    pub requires_password: bool,
    pub allow_late_join: bool,
    pub auto_close_on_empty: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_type: SessionType::ListeningParty,
            max_participants: 50,
            requires_password: false,
            allow_late_join: true,
            auto_close_on_empty: true,
        }
    }
}

/// Collaboration session
pub struct Session {
    pub id: Uuid,
    pub config: SessionConfig,
    state: RwLock<SessionState>,
    participants: RwLock<HashMap<Uuid, Participant>>,
    event_tx: broadcast::Sender<SessionEvent>,
}

/// Session events
#[derive(Debug, Clone)]
pub enum SessionEvent {
    ParticipantJoined { participant_id: Uuid },
    ParticipantLeft { participant_id: Uuid },
    StateChanged { old: SessionState, new: SessionState },
    Message { from: Uuid, content: String },
}

impl Session {
    pub fn new(config: SessionConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        Self {
            id: Uuid::new_v4(),
            config,
            state: RwLock::new(SessionState::Created),
            participants: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.event_tx.subscribe()
    }

    pub async fn start(&self) -> RealtimeResult<()> {
        let mut state = self.state.write().await;
        let old = *state;
        *state = SessionState::Active;
        let _ = self.event_tx.send(SessionEvent::StateChanged { old, new: SessionState::Active });
        Ok(())
    }

    pub async fn add_participant(&self, participant: Participant) -> RealtimeResult<()> {
        let state = self.state.read().await;
        if *state != SessionState::Active && *state != SessionState::Created {
            return Err(RealtimeError::SessionClosed);
        }
        drop(state);

        let mut participants = self.participants.write().await;
        if participants.len() >= self.config.max_participants {
            return Err(RealtimeError::SessionFull);
        }

        let participant_id = participant.id;
        participants.insert(participant_id, participant);
        let _ = self.event_tx.send(SessionEvent::ParticipantJoined { participant_id });
        Ok(())
    }

    pub async fn remove_participant(&self, participant_id: Uuid) -> RealtimeResult<()> {
        let mut participants = self.participants.write().await;
        participants.remove(&participant_id);
        let _ = self.event_tx.send(SessionEvent::ParticipantLeft { participant_id });

        // Auto-close if empty
        if participants.is_empty() && self.config.auto_close_on_empty {
            drop(participants);
            self.end().await?;
        }

        Ok(())
    }

    pub async fn get_participants(&self) -> Vec<Participant> {
        let participants = self.participants.read().await;
        participants.values().cloned().collect()
    }

    pub async fn end(&self) -> RealtimeResult<()> {
        let mut state = self.state.write().await;
        let old = *state;
        *state = SessionState::Ended;
        let _ = self.event_tx.send(SessionEvent::StateChanged { old, new: SessionState::Ended });
        Ok(())
    }

    pub async fn state(&self) -> SessionState {
        *self.state.read().await
    }
}

/// Session manager
pub struct SessionManager {
    sessions: RwLock<HashMap<Uuid, Arc<Session>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    pub async fn create_session(&self, config: SessionConfig) -> Arc<Session> {
        let session = Arc::new(Session::new(config));
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id, session.clone());
        session
    }

    pub async fn get_session(&self, session_id: Uuid) -> Option<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    pub async fn remove_session(&self, session_id: Uuid) -> Option<Arc<Session>> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(&session_id)
    }

    pub async fn list_sessions(&self) -> Vec<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
