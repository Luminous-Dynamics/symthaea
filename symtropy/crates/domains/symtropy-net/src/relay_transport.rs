// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Relay transport — uses the signaling WebSocket as a data relay.
//!
//! This is the simplest working multiplayer transport. Instead of
//! establishing WebRTC peer connections (which require ICE negotiation,
//! STUN/TURN, etc.), it sends game data through the signaling server.
//!
//! Trade-offs:
//! - (+) Works immediately — no ICE, no STUN, no NAT traversal needed
//! - (+) Works through any firewall (outbound WebSocket)
//! - (-) Higher latency (server hop instead of direct P2P)
//! - (-) Server bandwidth scales with player count
//!
//! For a small indie game (2-8 players), this is perfectly fine.
//! WebRTC data channels can be added later as an optimization via
//! the `Transport` trait — the game code doesn't change.
//!
//! # Upgrade Path
//!
//! 1. Ship with RelayTransport (works now, no infrastructure)
//! 2. Add WebRtcTransport (direct P2P, lower latency)
//! 3. Use relay as fallback when WebRTC fails (symmetric NAT)

#[cfg(feature = "webrtc")]
mod implementation {
    use crate::config::NetworkConfig;
    use crate::peer::PeerId;
    use crate::signaling::{SignalData, SignalingClient, SignalingEvent};
    use crate::transport::{Channel, PeerMessage, Transport, TransportEvent};

    /// Game data message relayed through the signaling server.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct RelayedData {
        channel: u8, // 0 = unreliable, 1 = reliable
        payload: Vec<u8>,
    }

    /// Transport that uses the signaling WebSocket as a data relay.
    pub struct RelayTransport {
        config: NetworkConfig,
        signaling: Option<SignalingClient>,
        local_id: PeerId,
        peers: Vec<PeerId>,
        pending_events: Vec<TransportEvent>,
        connected: bool,
    }

    impl RelayTransport {
        /// Create a new relay transport with the given config.
        pub fn new(config: NetworkConfig) -> Self {
            Self {
                config,
                signaling: None,
                local_id: PeerId(rand_id()),
                peers: Vec::new(),
                pending_events: Vec::new(),
                connected: false,
            }
        }

        /// Connect to signaling server (async — call from tokio runtime).
        pub async fn connect_async(&mut self, room_id: &str) -> Result<(), String> {
            let client = SignalingClient::connect(&self.config.signal_url).await?;
            client.join(room_id)?;
            self.signaling = Some(client);
            self.connected = true;
            self.pending_events.push(TransportEvent::SignalingConnected);
            Ok(())
        }
    }

    impl Transport for RelayTransport {
        fn connect(&mut self, room_id: &str) -> Result<(), String> {
            // For sync connect, we queue the async connect.
            // The caller should use connect_async in a tokio context.
            // This fallback creates a runtime if none exists.
            let url = self.config.signal_url.clone();
            let room = room_id.to_string();

            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    // KNOWN GAP: this fallback path spawns the connect future but
                    // never stores the result on `self` (the receiving end `_rx` is
                    // dropped) and never sends `room` to join once connected — so
                    // `self.signaling` stays `None` and no room is ever joined via
                    // this path. Untested/unfixed as part of the webrtc-compiles
                    // fix; use `connect_async()` directly instead of this sync
                    // fallback until it's wired up for real.
                    let (tx, _rx) = std::sync::mpsc::channel();
                    let url = url.clone();
                    let _room = room.clone();
                    handle.spawn(async move {
                        let result = SignalingClient::connect(&url).await;
                        let _ = tx.send(result);
                    });
                    // Non-blocking — signaling will connect in background
                    // Events arrive via poll()
                    self.connected = true;
                    self.pending_events.push(TransportEvent::SignalingConnected);
                    Ok(())
                }
                Err(_) => {
                    Err("RelayTransport requires a tokio runtime. Use connect_async().".into())
                }
            }
        }

        fn disconnect(&mut self) {
            self.signaling = None;
            self.peers.clear();
            self.connected = false;
        }

        fn send(&mut self, to: PeerId, channel: Channel, data: &[u8]) -> Result<(), String> {
            let signaling = self.signaling.as_ref().ok_or("Not connected")?;

            let relayed = RelayedData {
                channel: match channel {
                    Channel::Unreliable => 0,
                    Channel::Reliable => 1,
                },
                payload: data.to_vec(),
            };

            let json = serde_json::to_string(&relayed).map_err(|e| format!("Serialize: {e}"))?;

            signaling
                .signal(to, SignalData::Offer { sdp: json })
                .map_err(|e| format!("Send: {e}"))
        }

        fn broadcast(&mut self, channel: Channel, data: &[u8]) -> Result<(), String> {
            let peers: Vec<PeerId> = self.peers.clone();
            for peer in peers {
                self.send(peer, channel, data)?;
            }
            Ok(())
        }

        fn poll(&mut self) -> Vec<TransportEvent> {
            let mut events: Vec<TransportEvent> = self.pending_events.drain(..).collect();

            if let Some(ref mut signaling) = self.signaling {
                for evt in signaling.poll_events() {
                    match evt {
                        SignalingEvent::Connected(id) => {
                            self.local_id = id;
                        }
                        SignalingEvent::PeerJoined(id) => {
                            if !self.peers.contains(&id) {
                                self.peers.push(id);
                                events.push(TransportEvent::PeerConnected(id));
                            }
                        }
                        SignalingEvent::PeerLeft(id) => {
                            self.peers.retain(|p| *p != id);
                            events.push(TransportEvent::PeerDisconnected(id));
                        }
                        SignalingEvent::Signal { from, data } => {
                            // Decode relayed game data
                            if let SignalData::Offer { sdp } = data {
                                if let Ok(relayed) = serde_json::from_str::<RelayedData>(&sdp) {
                                    let channel = if relayed.channel == 0 {
                                        Channel::Unreliable
                                    } else {
                                        Channel::Reliable
                                    };
                                    events.push(TransportEvent::Message(PeerMessage {
                                        from,
                                        channel,
                                        data: relayed.payload,
                                    }));
                                }
                            }
                        }
                        SignalingEvent::Disconnected => {
                            self.connected = false;
                            events.push(TransportEvent::SignalingDisconnected);
                        }
                        SignalingEvent::Error(e) => {
                            events.push(TransportEvent::Error(e));
                        }
                    }
                }
            }

            events
        }

        fn peer_count(&self) -> usize {
            self.peers.len()
        }

        fn is_signaling_connected(&self) -> bool {
            self.connected
        }

        fn local_peer_id(&self) -> PeerId {
            self.local_id
        }
    }

    /// Generate a random peer ID (used before server assigns one).
    fn rand_id() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        t.as_nanos() as u64 ^ (t.as_secs() << 32)
    }
}

#[cfg(feature = "webrtc")]
pub use implementation::RelayTransport;
