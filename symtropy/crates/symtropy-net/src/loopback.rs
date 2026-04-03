// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Loopback transport — in-memory, zero-latency transport for testing.
//!
//! Two `LoopbackTransport` instances share a channel pair via `Arc<Mutex<>>`.
//! Messages sent by one appear instantly in the other's poll().
//! No signaling, no WebRTC — pure in-process testing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::peer::PeerId;
use crate::transport::{Channel, PeerMessage, Transport, TransportEvent};

/// Shared message queue between two loopback endpoints.
type SharedQueue = Arc<Mutex<VecDeque<PeerMessage>>>;

/// Create a connected pair of loopback transports.
///
/// Messages sent by `a` appear in `b.poll()` and vice versa.
pub fn loopback_pair() -> (LoopbackTransport, LoopbackTransport) {
    let a_to_b: SharedQueue = Arc::new(Mutex::new(VecDeque::new()));
    let b_to_a: SharedQueue = Arc::new(Mutex::new(VecDeque::new()));

    let a = LoopbackTransport {
        local_id: PeerId(0),
        remote_id: PeerId(1),
        outbox: a_to_b.clone(),
        inbox: b_to_a.clone(),
        connected: false,
    };

    let b = LoopbackTransport {
        local_id: PeerId(1),
        remote_id: PeerId(0),
        outbox: b_to_a,
        inbox: a_to_b,
        connected: false,
    };

    (a, b)
}

/// In-memory transport for testing P2P sync without real networking.
pub struct LoopbackTransport {
    local_id: PeerId,
    remote_id: PeerId,
    outbox: SharedQueue,
    inbox: SharedQueue,
    connected: bool,
}

impl Transport for LoopbackTransport {
    fn connect(&mut self, _room_id: &str) -> Result<(), String> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
    }

    fn send(&mut self, _to: PeerId, channel: Channel, data: &[u8]) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".into());
        }
        let msg = PeerMessage {
            from: self.local_id,
            channel,
            data: data.to_vec(),
        };
        self.outbox.lock().unwrap().push_back(msg);
        Ok(())
    }

    fn broadcast(&mut self, channel: Channel, data: &[u8]) -> Result<(), String> {
        self.send(self.remote_id, channel, data)
    }

    fn poll(&mut self) -> Vec<TransportEvent> {
        let mut events = Vec::new();

        if self.connected {
            let mut inbox = self.inbox.lock().unwrap();
            while let Some(msg) = inbox.pop_front() {
                events.push(TransportEvent::Message(msg));
            }
        }

        events
    }

    fn peer_count(&self) -> usize {
        if self.connected { 1 } else { 0 }
    }

    fn is_signaling_connected(&self) -> bool {
        self.connected
    }

    fn local_peer_id(&self) -> PeerId {
        self.local_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_send_receive() {
        let (mut a, mut b) = loopback_pair();
        a.connect("test").unwrap();
        b.connect("test").unwrap();

        a.send(PeerId(1), Channel::Reliable, b"hello").unwrap();

        let events = b.poll();
        assert_eq!(events.len(), 1);
        match &events[0] {
            TransportEvent::Message(msg) => {
                assert_eq!(msg.from, PeerId(0));
                assert_eq!(msg.channel, Channel::Reliable);
                assert_eq!(msg.data, b"hello");
            }
            _ => panic!("Expected Message event"),
        }
    }

    #[test]
    fn loopback_bidirectional() {
        let (mut a, mut b) = loopback_pair();
        a.connect("test").unwrap();
        b.connect("test").unwrap();

        a.broadcast(Channel::Unreliable, b"physics").unwrap();
        b.broadcast(Channel::Reliable, b"authority").unwrap();

        let b_events = b.poll();
        assert_eq!(b_events.len(), 1);

        let a_events = a.poll();
        assert_eq!(a_events.len(), 1);
        match &a_events[0] {
            TransportEvent::Message(msg) => {
                assert_eq!(msg.from, PeerId(1));
                assert_eq!(msg.data, b"authority");
            }
            _ => panic!("Expected Message"),
        }
    }

    #[test]
    fn loopback_not_connected() {
        let (mut a, _b) = loopback_pair();
        assert!(a.send(PeerId(1), Channel::Reliable, b"fail").is_err());
    }

    #[test]
    fn loopback_peer_count() {
        let (mut a, _b) = loopback_pair();
        assert_eq!(a.peer_count(), 0);
        a.connect("test").unwrap();
        assert_eq!(a.peer_count(), 1);
        a.disconnect();
        assert_eq!(a.peer_count(), 0);
    }
}
