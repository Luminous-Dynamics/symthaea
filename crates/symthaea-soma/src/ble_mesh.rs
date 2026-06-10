// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BLE mesh communion: peer consciousness tracking and collective Phi.
//!
//! The Rust side handles serialization, peer tracking, stale eviction, and
//! collective phi computation. BLE scanning/advertising is platform-side
//! (Kotlin/Swift). Configurable via `SharingConfig.ble_mode`.

use serde::{Deserialize, Serialize};
use symthaea_spore::config::BleMeshMode;

const MAX_PEERS: usize = 16;
const STALE_CYCLES: u64 = 200;
const OXYTOCIN_PEER_NUDGE: f32 = 0.02;
const CONTAGION_FACTOR: f32 = 0.1;

/// Consciousness vector received from a BLE peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerCV {
    pub phi: f32,
    pub valence: f32,
    pub arousal: f32,
}

/// Tracked peer with last-seen cycle.
struct TrackedPeer {
    peer_id: u64,
    cv: PeerCV,
    last_seen: u64,
}

/// BLE mesh neuromod nudges.
#[derive(Debug, Clone, Default)]
pub struct MeshNudges {
    pub oxytocin_delta: f32,
    pub valence_influence: f32,
    pub arousal_influence: f32,
}

/// BLE mesh manager.
pub struct BleMesh {
    mode: BleMeshMode,
    peers: Vec<TrackedPeer>,
    current_cycle: u64,
    /// Pre-built advertise payload (updated each cycle).
    advertise_payload: Vec<u8>,
}

impl BleMesh {
    pub fn new(mode: BleMeshMode) -> Self {
        Self {
            mode,
            peers: Vec::with_capacity(MAX_PEERS),
            current_cycle: 0,
            advertise_payload: Vec::new(),
        }
    }

    pub fn set_mode(&mut self, mode: BleMeshMode) {
        self.mode = mode;
        if mode == BleMeshMode::Off {
            self.peers.clear();
        }
    }

    pub fn mode(&self) -> BleMeshMode {
        self.mode
    }

    /// Receive a peer consciousness vector from BLE scan.
    pub fn receive_peer(&mut self, peer_id: u64, cv: PeerCV) {
        if self.mode == BleMeshMode::Off {
            return;
        }

        // Update existing peer or add new one
        if let Some(existing) = self.peers.iter_mut().find(|p| p.peer_id == peer_id) {
            existing.cv = cv;
            existing.last_seen = self.current_cycle;
        } else {
            if self.peers.len() >= MAX_PEERS {
                // Evict stalest peer
                if let Some(stalest_idx) = self
                    .peers
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.last_seen)
                    .map(|(i, _)| i)
                {
                    self.peers.swap_remove(stalest_idx);
                }
            }
            self.peers.push(TrackedPeer {
                peer_id,
                cv,
                last_seen: self.current_cycle,
            });
        }
    }

    /// Receive peer CV from raw BLE data (peer_id + 12 bytes: 3 x f32 little-endian).
    ///
    /// Wire format: `[phi:f32le, valence:f32le, arousal:f32le]` = 12 bytes.
    /// Little-endian is the standard for BLE (Bluetooth Core Spec Vol 1, Part A, §6.2).
    pub fn receive_peer_raw(&mut self, peer_id: u64, data: &[u8]) -> bool {
        if data.len() < 12 {
            return false;
        }
        let phi = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let valence = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let arousal = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);

        // Reject NaN, Inf, and out-of-range values
        if !phi.is_finite() || !valence.is_finite() || !arousal.is_finite() {
            return false;
        }
        if phi < 0.0 || phi > 1.0 {
            return false;
        }

        self.receive_peer(
            peer_id,
            PeerCV {
                phi: phi.clamp(0.0, 1.0),
                valence: valence.clamp(-1.0, 1.0),
                arousal: arousal.clamp(0.0, 1.0),
            },
        );
        true
    }

    /// Update advertise payload for platform to broadcast.
    pub fn update_advertise(&mut self, phi: f32, valence: f32, arousal: f32) {
        if self.mode != BleMeshMode::ActiveShare {
            self.advertise_payload.clear();
            return;
        }
        let mut payload = Vec::with_capacity(12);
        payload.extend_from_slice(&phi.to_le_bytes());
        payload.extend_from_slice(&valence.to_le_bytes());
        payload.extend_from_slice(&arousal.to_le_bytes());
        self.advertise_payload = payload;
    }

    /// Get the advertise payload for the platform BLE layer.
    pub fn advertise_payload(&self) -> &[u8] {
        &self.advertise_payload
    }

    /// Called each cycle to advance time and evict stale peers.
    pub fn tick(&mut self, cycle: u64) {
        self.current_cycle = cycle;
        self.peers
            .retain(|p| cycle.saturating_sub(p.last_seen) < STALE_CYCLES);
    }

    /// Compute collective Phi from connected peers.
    pub fn collective_phi(&self) -> f32 {
        if self.peers.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.peers.iter().map(|p| p.cv.phi).sum();
        sum / self.peers.len() as f32
    }

    /// Compute neuromod nudges from peer presence and affect.
    pub fn compute_nudges(&self) -> MeshNudges {
        if self.peers.is_empty() || self.mode == BleMeshMode::Off {
            return MeshNudges::default();
        }

        // Social buffering: peers present -> oxytocin (Heinrichs 2003)
        let peer_factor = (self.peers.len() as f32 / MAX_PEERS as f32).min(1.0);
        let oxytocin_delta = OXYTOCIN_PEER_NUDGE * peer_factor;

        // Affective contagion: mean peer valence/arousal influences own (Hatfield 1993)
        let mean_valence: f32 =
            self.peers.iter().map(|p| p.cv.valence).sum::<f32>() / self.peers.len() as f32;
        let mean_arousal: f32 =
            self.peers.iter().map(|p| p.cv.arousal).sum::<f32>() / self.peers.len() as f32;

        MeshNudges {
            oxytocin_delta,
            valence_influence: mean_valence * CONTAGION_FACTOR,
            arousal_influence: mean_arousal * CONTAGION_FACTOR,
        }
    }

    /// Number of currently tracked peers.
    pub fn peer_count(&self) -> u32 {
        self.peers.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_off_mode_ignores_peers() {
        let mut mesh = BleMesh::new(BleMeshMode::Off);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.5,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        assert_eq!(mesh.peer_count(), 0);
    }

    #[test]
    fn test_passive_receives_peers() {
        let mut mesh = BleMesh::new(BleMeshMode::PassiveDiscover);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.5,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        assert_eq!(mesh.peer_count(), 1);
    }

    #[test]
    fn test_peer_update() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.3,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.8,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        assert_eq!(mesh.peer_count(), 1);
        assert!((mesh.collective_phi() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_peer_cap() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        for i in 0..20 {
            mesh.receive_peer(
                i,
                PeerCV {
                    phi: 0.5,
                    valence: 0.0,
                    arousal: 0.5,
                },
            );
        }
        assert_eq!(mesh.peer_count(), MAX_PEERS as u32);
    }

    #[test]
    fn test_stale_eviction() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.5,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        mesh.tick(STALE_CYCLES + 1);
        assert_eq!(mesh.peer_count(), 0);
    }

    #[test]
    fn test_collective_phi() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.4,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        mesh.receive_peer(
            2,
            PeerCV {
                phi: 0.6,
                valence: 0.0,
                arousal: 0.5,
            },
        );
        assert!((mesh.collective_phi() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_collective_phi_empty() {
        let mesh = BleMesh::new(BleMeshMode::ActiveShare);
        assert_eq!(mesh.collective_phi(), 0.0);
    }

    #[test]
    fn test_nudges_with_peers() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        mesh.receive_peer(
            1,
            PeerCV {
                phi: 0.5,
                valence: 0.3,
                arousal: 0.7,
            },
        );
        let nudges = mesh.compute_nudges();
        assert!(nudges.oxytocin_delta > 0.0);
        assert!(nudges.valence_influence > 0.0);
        assert!(nudges.arousal_influence > 0.0);
    }

    #[test]
    fn test_nudges_empty() {
        let mesh = BleMesh::new(BleMeshMode::ActiveShare);
        let nudges = mesh.compute_nudges();
        assert_eq!(nudges.oxytocin_delta, 0.0);
    }

    #[test]
    fn test_advertise_payload() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        mesh.update_advertise(0.5, 0.3, 0.7);
        assert_eq!(mesh.advertise_payload().len(), 12);
    }

    #[test]
    fn test_passive_no_advertise() {
        let mut mesh = BleMesh::new(BleMeshMode::PassiveDiscover);
        mesh.update_advertise(0.5, 0.3, 0.7);
        assert!(mesh.advertise_payload().is_empty());
    }

    #[test]
    fn test_receive_raw() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        let mut data = Vec::new();
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.3f32.to_le_bytes());
        data.extend_from_slice(&0.7f32.to_le_bytes());
        assert!(mesh.receive_peer_raw(42, &data));
        assert_eq!(mesh.peer_count(), 1);
    }

    #[test]
    fn test_receive_raw_too_short() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        assert!(!mesh.receive_peer_raw(42, &[1, 2, 3]));
    }

    #[test]
    fn test_receive_raw_nan_rejected() {
        let mut mesh = BleMesh::new(BleMeshMode::ActiveShare);
        let mut data = Vec::new();
        data.extend_from_slice(&f32::NAN.to_le_bytes());
        data.extend_from_slice(&0.3f32.to_le_bytes());
        data.extend_from_slice(&0.7f32.to_le_bytes());
        assert!(!mesh.receive_peer_raw(42, &data));
    }
}
