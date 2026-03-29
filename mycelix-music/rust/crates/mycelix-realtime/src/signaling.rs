// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebRTC signaling server

use crate::{RealtimeError, RealtimeResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

/// Signaling message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignalingMessage {
    Offer { sdp: String },
    Answer { sdp: String },
    IceCandidate { candidate: String, sdp_mid: Option<String>, sdp_m_line_index: Option<u32> },
    Join { room_id: Uuid },
    Leave { room_id: Uuid },
    Ping,
    Pong,
}

/// Signaling peer
#[derive(Debug, Clone)]
pub struct SignalingPeer {
    pub id: Uuid,
    pub room_id: Option<Uuid>,
    pub display_name: String,
}

/// Signaling server for WebRTC connections
pub struct SignalingServer {
    peers: RwLock<HashMap<Uuid, SignalingPeer>>,
    rooms: RwLock<HashMap<Uuid, Vec<Uuid>>>,
    message_tx: broadcast::Sender<(Uuid, SignalingMessage)>,
}

impl SignalingServer {
    pub fn new() -> Self {
        let (message_tx, _) = broadcast::channel(1000);
        Self {
            peers: RwLock::new(HashMap::new()),
            rooms: RwLock::new(HashMap::new()),
            message_tx,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<(Uuid, SignalingMessage)> {
        self.message_tx.subscribe()
    }

    pub async fn register_peer(&self, peer: SignalingPeer) -> RealtimeResult<()> {
        let mut peers = self.peers.write().await;
        peers.insert(peer.id, peer);
        Ok(())
    }

    pub async fn unregister_peer(&self, peer_id: Uuid) -> RealtimeResult<()> {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.remove(&peer_id) {
            if let Some(room_id) = peer.room_id {
                let mut rooms = self.rooms.write().await;
                if let Some(room) = rooms.get_mut(&room_id) {
                    room.retain(|id| *id != peer_id);
                }
            }
        }
        Ok(())
    }

    pub async fn join_room(&self, peer_id: Uuid, room_id: Uuid) -> RealtimeResult<Vec<Uuid>> {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.get_mut(&peer_id) {
            peer.room_id = Some(room_id);
        } else {
            return Err(RealtimeError::PeerNotFound(peer_id));
        }
        drop(peers);

        let mut rooms = self.rooms.write().await;
        let room = rooms.entry(room_id).or_insert_with(Vec::new);
        let existing_peers: Vec<Uuid> = room.iter().cloned().collect();
        room.push(peer_id);

        let _ = self.message_tx.send((peer_id, SignalingMessage::Join { room_id }));

        Ok(existing_peers)
    }

    pub async fn leave_room(&self, peer_id: Uuid) -> RealtimeResult<()> {
        let mut peers = self.peers.write().await;
        let room_id = if let Some(peer) = peers.get_mut(&peer_id) {
            let room_id = peer.room_id;
            peer.room_id = None;
            room_id
        } else {
            return Ok(());
        };
        drop(peers);

        if let Some(room_id) = room_id {
            let mut rooms = self.rooms.write().await;
            if let Some(room) = rooms.get_mut(&room_id) {
                room.retain(|id| *id != peer_id);
            }
            let _ = self.message_tx.send((peer_id, SignalingMessage::Leave { room_id }));
        }

        Ok(())
    }

    pub async fn send_to_peer(&self, from: Uuid, to: Uuid, message: SignalingMessage) -> RealtimeResult<()> {
        let peers = self.peers.read().await;
        if !peers.contains_key(&to) {
            return Err(RealtimeError::PeerNotFound(to));
        }
        drop(peers);

        let _ = self.message_tx.send((to, message));
        Ok(())
    }

    pub async fn broadcast_to_room(&self, from: Uuid, room_id: Uuid, message: SignalingMessage) -> RealtimeResult<()> {
        let rooms = self.rooms.read().await;
        let room = rooms.get(&room_id).cloned().unwrap_or_default();
        drop(rooms);

        for peer_id in room {
            if peer_id != from {
                let _ = self.message_tx.send((peer_id, message.clone()));
            }
        }

        Ok(())
    }

    pub async fn get_room_peers(&self, room_id: Uuid) -> Vec<Uuid> {
        let rooms = self.rooms.read().await;
        rooms.get(&room_id).cloned().unwrap_or_default()
    }
}

impl Default for SignalingServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple offer/answer exchange helper
pub struct OfferAnswerExchange {
    pub peer_a: Uuid,
    pub peer_b: Uuid,
    pub offer: Option<String>,
    pub answer: Option<String>,
    pub ice_candidates_a: Vec<String>,
    pub ice_candidates_b: Vec<String>,
}

impl OfferAnswerExchange {
    pub fn new(peer_a: Uuid, peer_b: Uuid) -> Self {
        Self {
            peer_a,
            peer_b,
            offer: None,
            answer: None,
            ice_candidates_a: Vec::new(),
            ice_candidates_b: Vec::new(),
        }
    }

    pub fn set_offer(&mut self, offer: String) {
        self.offer = Some(offer);
    }

    pub fn set_answer(&mut self, answer: String) {
        self.answer = Some(answer);
    }

    pub fn add_ice_candidate(&mut self, from: Uuid, candidate: String) {
        if from == self.peer_a {
            self.ice_candidates_a.push(candidate);
        } else if from == self.peer_b {
            self.ice_candidates_b.push(candidate);
        }
    }

    pub fn is_complete(&self) -> bool {
        self.offer.is_some() && self.answer.is_some()
    }
}
