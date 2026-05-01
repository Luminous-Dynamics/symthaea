// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial authority partitioning for P2P physics.
//!
//! Each peer owns the physics for entities within their authority radius.
//! Entities outside all authority zones use the shared deterministic simulation.

use std::collections::{HashMap, HashSet};

use symtropy_physics::body::BodyHandle;

use crate::peer::PeerId;

/// Spatial authority system: determines which peer computes physics for which bodies.
pub struct SpatialAuthority {
    /// Which peer has authority over each body.
    body_authority: HashMap<BodyHandle, PeerId>,
    /// Bodies owned by the local peer.
    local_bodies: HashSet<BodyHandle>,
    /// Authority radius (bodies within this distance of the peer's player are owned).
    pub authority_radius: f64,
    /// Local peer ID.
    pub local_peer: PeerId,
}

impl SpatialAuthority {
    /// Create a new spatial authority system.
    pub fn new(local_peer: PeerId, authority_radius: f64) -> Self {
        Self {
            body_authority: HashMap::new(),
            local_bodies: HashSet::new(),
            authority_radius,
            local_peer,
        }
    }

    /// Claim authority over a body.
    pub fn claim(&mut self, body: BodyHandle, peer: PeerId) {
        self.body_authority.insert(body, peer);
        if peer == self.local_peer {
            self.local_bodies.insert(body);
        } else {
            self.local_bodies.remove(&body);
        }
    }

    /// Release authority over a body.
    pub fn release(&mut self, body: BodyHandle) {
        if let Some(peer) = self.body_authority.remove(&body) {
            if peer == self.local_peer {
                self.local_bodies.remove(&body);
            }
        }
    }

    /// Whether the local peer has authority over a body.
    pub fn is_local(&self, body: BodyHandle) -> bool {
        self.local_bodies.contains(&body)
    }

    /// Which peer has authority over a body.
    pub fn authority_of(&self, body: BodyHandle) -> Option<PeerId> {
        self.body_authority.get(&body).copied()
    }

    /// All bodies the local peer has authority over.
    pub fn local_body_count(&self) -> usize {
        self.local_bodies.len()
    }

    /// All bodies with assigned authority.
    pub fn total_claimed(&self) -> usize {
        self.body_authority.len()
    }

    /// Update authority based on distances from peers' players.
    /// `peer_positions` maps PeerId to (player_body, player_position_1d_dist).
    /// A body goes to the nearest peer within authority_radius.
    pub fn update_from_distances(
        &mut self,
        bodies: &[BodyHandle],
        body_distances_to_local: &HashMap<BodyHandle, f64>,
        remote_peer_claims: &HashMap<BodyHandle, PeerId>,
    ) {
        for &body in bodies {
            let local_dist = body_distances_to_local
                .get(&body)
                .copied()
                .unwrap_or(f64::MAX);

            if local_dist < self.authority_radius {
                // Check if a remote peer has a stronger claim
                if let Some(&remote_peer) = remote_peer_claims.get(&body) {
                    // Remote peer already claimed — don't steal
                    // In a real implementation, the closer peer wins
                    self.claim(body, remote_peer);
                } else {
                    self.claim(body, self.local_peer);
                }
            } else if let Some(&remote_peer) = remote_peer_claims.get(&body) {
                self.claim(body, remote_peer);
            } else {
                // No one claims this body — use shared simulation
                self.release(body);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_and_check() {
        let mut auth = SpatialAuthority::new(PeerId(0), 100.0);
        let body = BodyHandle(1);

        auth.claim(body, PeerId(0));
        assert!(auth.is_local(body));
        assert_eq!(auth.authority_of(body), Some(PeerId(0)));
    }

    #[test]
    fn remote_claim() {
        let mut auth = SpatialAuthority::new(PeerId(0), 100.0);
        let body = BodyHandle(1);

        auth.claim(body, PeerId(5));
        assert!(!auth.is_local(body));
        assert_eq!(auth.authority_of(body), Some(PeerId(5)));
    }

    #[test]
    fn release_clears() {
        let mut auth = SpatialAuthority::new(PeerId(0), 100.0);
        let body = BodyHandle(1);

        auth.claim(body, PeerId(0));
        assert!(auth.is_local(body));

        auth.release(body);
        assert!(!auth.is_local(body));
        assert_eq!(auth.authority_of(body), None);
    }

    #[test]
    fn local_body_count() {
        let mut auth = SpatialAuthority::new(PeerId(0), 100.0);
        auth.claim(BodyHandle(1), PeerId(0));
        auth.claim(BodyHandle(2), PeerId(0));
        auth.claim(BodyHandle(3), PeerId(1)); // remote
        assert_eq!(auth.local_body_count(), 2);
        assert_eq!(auth.total_claimed(), 3);
    }

    #[test]
    fn distance_based_update() {
        let mut auth = SpatialAuthority::new(PeerId(0), 50.0);

        let bodies = vec![BodyHandle(1), BodyHandle(2), BodyHandle(3)];
        let mut distances = HashMap::new();
        distances.insert(BodyHandle(1), 10.0); // close → local
        distances.insert(BodyHandle(2), 30.0); // close → local
        distances.insert(BodyHandle(3), 100.0); // far → unclaimed

        let remote_claims = HashMap::new();
        auth.update_from_distances(&bodies, &distances, &remote_claims);

        assert!(auth.is_local(BodyHandle(1)));
        assert!(auth.is_local(BodyHandle(2)));
        assert!(!auth.is_local(BodyHandle(3)));
    }

    #[test]
    fn remote_peer_takes_priority() {
        let mut auth = SpatialAuthority::new(PeerId(0), 50.0);

        let bodies = vec![BodyHandle(1)];
        let mut distances = HashMap::new();
        distances.insert(BodyHandle(1), 20.0); // within our radius

        let mut remote_claims = HashMap::new();
        remote_claims.insert(BodyHandle(1), PeerId(2)); // but remote claimed it

        auth.update_from_distances(&bodies, &distances, &remote_claims);
        assert!(!auth.is_local(BodyHandle(1)));
        assert_eq!(auth.authority_of(BodyHandle(1)), Some(PeerId(2)));
    }

    #[test]
    fn transfer_authority() {
        let mut auth = SpatialAuthority::new(PeerId(0), 50.0);
        let body = BodyHandle(1);

        // Initially local
        auth.claim(body, PeerId(0));
        assert!(auth.is_local(body));

        // Transfer to remote
        auth.claim(body, PeerId(3));
        assert!(!auth.is_local(body));
        assert_eq!(auth.authority_of(body), Some(PeerId(3)));
    }
}
