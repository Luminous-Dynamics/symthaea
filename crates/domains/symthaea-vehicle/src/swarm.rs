// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm mesh simulator: peer vehicles broadcasting Wisdom Vectors.
//!
//! Provides `SwarmSimulator` which manages N peer vehicles on a shared road,
//! stepping them with PD cruise control and computing the 4 mesh channels
//! (nearest_peer_distance, peer_relative_speed, mesh_brake_density, mesh_confidence)
//! in the ego vehicle's body frame.

use std::collections::VecDeque;

use crate::road::Road;
use crate::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use crate::types::{MeshSignal, VehicleCommand, VehicleState, pd_cruise_baseline_with_road};

/// Configuration for a swarm simulation.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Number of peer vehicles (excluding ego).
    pub num_peers: usize,
    /// Initial spacing between peers along the road (m).
    pub initial_spacing: f64,
    /// Road model shared by all peers.
    pub road: Road,
    /// Target cruising speed (m/s).
    pub target_speed: f64,
    /// Simulated mesh radio latency (s). Physically enforced: the ego sees
    /// each peer's broadcast state from `mesh_latency` seconds ago, not its
    /// live state (V2V was previously omniscient/instant).
    pub mesh_latency: f64,
    /// Confidence decay rate per meter of distance.
    pub confidence_decay_per_meter: f64,
}

impl SwarmConfig {
    /// Convoy on a straight road with given spacing.
    pub fn convoy_straight(num_peers: usize) -> Self {
        Self {
            num_peers,
            initial_spacing: 30.0,
            road: Road::straight(),
            target_speed: 13.4,
            mesh_latency: 0.01,
            confidence_decay_per_meter: 0.005,
        }
    }

    /// Convoy on a gentle highway with wider spacing.
    pub fn convoy_highway(num_peers: usize) -> Self {
        Self {
            num_peers,
            initial_spacing: 40.0,
            road: Road::gentle_highway(),
            target_speed: 13.4,
            mesh_latency: 0.01,
            confidence_decay_per_meter: 0.005,
        }
    }
}

/// A single peer vehicle in the swarm.
pub struct PeerVehicle {
    /// Physics simulator for this peer.
    pub sim: BicycleModelSimulator,
    /// Unique peer identifier.
    pub id: u64,
    /// Whether this peer is currently hard-braking (scenario trigger).
    pub is_braking: bool,
    /// Ring of (sim_time, state) broadcast snapshots — what the radio has
    /// sent so far. The ego reads from this with `mesh_latency` delay.
    history: VecDeque<(f64, VehicleState)>,
}

impl PeerVehicle {
    /// The peer state as seen over the mesh at `now`: the newest snapshot
    /// that is at least `latency` old. Before any snapshot is old enough
    /// (start-up), the oldest available one is used — a radio that has
    /// broadcast anything reports its earliest frame, never the live state.
    fn broadcast_state(&self, now: f64, latency: f64) -> &VehicleState {
        let cutoff = now - latency;
        self.history
            .iter()
            .rev()
            .find(|(t, _)| *t <= cutoff)
            .or_else(|| self.history.front())
            .map(|(_, s)| s)
            .unwrap_or_else(|| self.sim.state())
    }
}

/// Swarm simulator managing N peer vehicles.
pub struct SwarmSimulator {
    config: SwarmConfig,
    peers: Vec<PeerVehicle>,
    /// Accumulated simulation time (s) — drives the latency buffer.
    sim_time: f64,
}

impl SwarmSimulator {
    /// Create a new swarm with peers staggered along the road ahead of the origin.
    pub fn new(config: SwarmConfig) -> Self {
        let mut peers = Vec::with_capacity(config.num_peers);

        for i in 0..config.num_peers {
            let mut sim = BicycleModelSimulator::at_speed(config.target_speed);
            // Stagger peers ahead along the road: peer 0 is closest, peer N-1 is farthest
            let s = (i as f64 + 1.0) * config.initial_spacing;
            let (x, y, heading) = config.road.position_at(s);
            sim.set_position(x, y, heading);
            // Seed the broadcast history with the initial state so the
            // mesh is never empty at t=0.
            let initial = sim.state().clone();
            let mut history = VecDeque::new();
            history.push_back((0.0, initial));
            peers.push(PeerVehicle {
                sim,
                id: i as u64 + 1,
                is_braking: false,
                history,
            });
        }

        Self {
            config,
            peers,
            sim_time: 0.0,
        }
    }

    /// Number of peers.
    pub fn num_peers(&self) -> usize {
        self.peers.len()
    }

    /// Step all peer vehicles using PD cruise control on the road.
    pub fn step(&mut self, dt: f64) {
        for peer in &mut self.peers {
            let state = peer.sim.state().clone();
            let proj = self.config.road.project(state.position_x, state.position_y);

            let cmd = if peer.is_braking {
                VehicleCommand {
                    steering: 0.0,
                    throttle: 0.0,
                    brake: 1.0,
                }
            } else {
                pd_cruise_baseline_with_road(
                    peer.sim.state(),
                    self.config.target_speed,
                    proj.road_heading,
                )
            };

            peer.sim.step(&cmd, dt);
        }

        // Record this step's broadcasts and drop frames nobody can still
        // receive (older than the latency window, plus one frame of slack).
        self.sim_time += dt;
        let prune_before = self.sim_time - self.config.mesh_latency - 2.0 * dt;
        for peer in &mut self.peers {
            peer.history
                .push_back((self.sim_time, peer.sim.state().clone()));
            while peer
                .history
                .front()
                .is_some_and(|(t, _)| *t < prune_before && peer.history.len() > 1)
            {
                peer.history.pop_front();
            }
        }
    }

    /// Force a peer to start hard-braking (for scenario testing).
    pub fn trigger_lead_brake(&mut self, peer_idx: usize) {
        if let Some(peer) = self.peers.get_mut(peer_idx) {
            peer.is_braking = true;
        }
    }

    /// Reset all peers to non-braking state.
    pub fn reset_braking(&mut self) {
        for peer in &mut self.peers {
            peer.is_braking = false;
        }
    }

    /// Compute mesh signals from all peers relative to the ego vehicle.
    ///
    /// Peer states are read through the latency buffer — the ego sees each
    /// peer's broadcast from `mesh_latency` seconds ago, not its live state.
    pub fn mesh_signals(&self, ego_state: &VehicleState) -> Vec<MeshSignal> {
        self.peers
            .iter()
            .map(|peer| {
                let ps = peer.broadcast_state(self.sim_time, self.config.mesh_latency);
                let dx = ps.position_x - ego_state.position_x;
                let dy = ps.position_y - ego_state.position_y;
                let dist = (dx * dx + dy * dy).sqrt();

                // Transform to ego body frame
                let cos_h = ego_state.heading.cos();
                let sin_h = ego_state.heading.sin();
                let lon = dx * cos_h + dy * sin_h;
                let lat = -dx * sin_h + dy * cos_h;

                let confidence =
                    (1.0 - dist * self.config.confidence_decay_per_meter).clamp(0.0, 1.0);

                MeshSignal {
                    peer_id: peer.id,
                    relative_position: [lon, lat],
                    peer_speed: ps.speed,
                    peer_yaw_rate: ps.yaw_rate,
                    peer_braking: ps.brake_pressure > 0.3,
                    peer_abs_engaged: ps.brake_pressure > 0.8,
                    latency: self.config.mesh_latency,
                    confidence,
                }
            })
            .collect()
    }

    /// Compute the 4 aggregate mesh channels and write them onto the ego state.
    pub fn aggregate_mesh(&self, ego_state: &mut VehicleState) {
        if self.peers.is_empty() {
            ego_state.nearest_peer_distance = 200.0;
            ego_state.peer_relative_speed = 0.0;
            ego_state.mesh_brake_density = 0.0;
            ego_state.mesh_confidence = 0.0;
            return;
        }

        let signals = self.mesh_signals(ego_state);

        // Nearest peer distance
        let mut min_dist = f64::MAX;
        let mut nearest_speed = 0.0;
        let mut brake_count = 0usize;
        let mut total_confidence = 0.0;

        for sig in &signals {
            let dist = (sig.relative_position[0].powi(2) + sig.relative_position[1].powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_speed = sig.peer_speed;
            }
            if sig.peer_braking {
                brake_count += 1;
            }
            total_confidence += sig.confidence;
        }

        ego_state.nearest_peer_distance = min_dist.min(200.0);
        ego_state.peer_relative_speed = nearest_speed - ego_state.speed;
        ego_state.mesh_brake_density = brake_count as f64 / signals.len() as f64;
        ego_state.mesh_confidence = total_confidence / signals.len() as f64;
    }

    /// Access peers (read-only).
    pub fn peers(&self) -> &[PeerVehicle] {
        &self.peers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_creation() {
        let config = SwarmConfig::convoy_straight(3);
        let swarm = SwarmSimulator::new(config);
        assert_eq!(swarm.num_peers(), 3);
        // Peers should be ahead of origin
        for peer in swarm.peers() {
            assert!(
                peer.sim.state().position_x > 10.0,
                "Peer should be ahead: x={}",
                peer.sim.state().position_x
            );
        }
    }

    #[test]
    fn test_step_advances_peers() {
        let config = SwarmConfig::convoy_straight(2);
        let mut swarm = SwarmSimulator::new(config);
        let x0 = swarm.peers()[0].sim.state().position_x;
        for _ in 0..100 {
            swarm.step(0.005);
        }
        let x1 = swarm.peers()[0].sim.state().position_x;
        assert!(x1 > x0, "Peer should advance: {x0} → {x1}");
    }

    #[test]
    fn test_nearest_peer_distance() {
        let config = SwarmConfig::convoy_straight(2);
        let swarm = SwarmSimulator::new(config);
        let mut ego = VehicleState::cruising(13.4);
        swarm.aggregate_mesh(&mut ego);
        // First peer is at ~30m ahead
        assert!(
            ego.nearest_peer_distance < 40.0,
            "Nearest should be close: {}",
            ego.nearest_peer_distance
        );
        assert!(
            ego.nearest_peer_distance > 20.0,
            "Nearest should not be too close: {}",
            ego.nearest_peer_distance
        );
    }

    #[test]
    fn test_brake_density() {
        let config = SwarmConfig::convoy_straight(4);
        let mut swarm = SwarmSimulator::new(config);
        // No braking initially
        let mut ego = VehicleState::cruising(13.4);
        swarm.aggregate_mesh(&mut ego);
        assert!(
            ego.mesh_brake_density < 0.01,
            "No braking: {}",
            ego.mesh_brake_density
        );

        // Trigger 2 of 4 peers to brake
        swarm.trigger_lead_brake(0);
        swarm.trigger_lead_brake(1);
        // Step to let brake pressure register
        for _ in 0..100 {
            swarm.step(0.005);
        }
        swarm.aggregate_mesh(&mut ego);
        assert!(
            ego.mesh_brake_density > 0.4,
            "Should have ~50% braking: {}",
            ego.mesh_brake_density
        );
    }

    #[test]
    fn test_mesh_latency_delays_brake_propagation() {
        // A peer that starts braking must NOT be visible as braking over
        // the mesh until mesh_latency has elapsed — V2V used to read live
        // peer state (omniscient/instant).
        let mut config = SwarmConfig::convoy_straight(1);
        config.mesh_latency = 0.5;
        let mut swarm = SwarmSimulator::new(config);
        let ego = VehicleState::cruising(13.4);
        let dt = 0.005;

        // Build some pre-braking history.
        for _ in 0..10 {
            swarm.step(dt);
        }
        swarm.trigger_lead_brake(0);
        // 0.05 s after braking starts: live state brakes, mesh must not
        // see it yet (delayed view is 0.5 s old).
        for _ in 0..10 {
            swarm.step(dt);
        }
        assert!(
            swarm.peers()[0].sim.state().brake_pressure > 0.3,
            "precondition: live peer must actually be braking"
        );
        let signals = swarm.mesh_signals(&ego);
        assert!(
            !signals[0].peer_braking,
            "mesh must not see the brake before mesh_latency elapses"
        );

        // 0.65 s after braking starts: the delayed view now includes it.
        for _ in 0..120 {
            swarm.step(dt);
        }
        let signals = swarm.mesh_signals(&ego);
        assert!(
            signals[0].peer_braking,
            "mesh must see the brake after mesh_latency elapses"
        );
    }

    #[test]
    fn test_zero_mesh_latency_is_live() {
        let mut config = SwarmConfig::convoy_straight(1);
        config.mesh_latency = 0.0;
        let mut swarm = SwarmSimulator::new(config);
        let ego = VehicleState::cruising(13.4);
        swarm.trigger_lead_brake(0);
        swarm.step(0.005);
        let signals = swarm.mesh_signals(&ego);
        assert!(
            signals[0].peer_braking,
            "zero latency must behave like the old live read"
        );
    }

    #[test]
    fn test_confidence_decay() {
        let mut config = SwarmConfig::convoy_straight(1);
        config.initial_spacing = 50.0;
        config.confidence_decay_per_meter = 0.01;
        let swarm = SwarmSimulator::new(config);
        let ego = VehicleState::cruising(13.4);
        let signals = swarm.mesh_signals(&ego);
        assert_eq!(signals.len(), 1);
        // Peer at ~50m: confidence = 1 - 50*0.01 = 0.5
        assert!(
            signals[0].confidence < 0.7,
            "Should decay: {}",
            signals[0].confidence
        );
        assert!(
            signals[0].confidence > 0.3,
            "Should not be zero: {}",
            signals[0].confidence
        );
    }

    #[test]
    fn test_no_peers_defaults() {
        let mut config = SwarmConfig::convoy_straight(0);
        config.num_peers = 0;
        let swarm = SwarmSimulator::new(config);
        let mut ego = VehicleState::cruising(13.4);
        swarm.aggregate_mesh(&mut ego);
        assert!(
            (ego.nearest_peer_distance - 200.0).abs() < 0.01,
            "No peers: {}",
            ego.nearest_peer_distance
        );
        assert!(
            ego.mesh_brake_density.abs() < 0.01,
            "No braking: {}",
            ego.mesh_brake_density
        );
        assert!(
            ego.mesh_confidence.abs() < 0.01,
            "No confidence: {}",
            ego.mesh_confidence
        );
    }

    #[test]
    fn test_lead_brake_scenario() {
        let config = SwarmConfig::convoy_straight(3);
        let mut swarm = SwarmSimulator::new(config);
        swarm.trigger_lead_brake(0);
        // Step for braking to take effect
        for _ in 0..200 {
            swarm.step(0.005);
        }
        let mut ego = VehicleState::cruising(13.4);
        swarm.aggregate_mesh(&mut ego);
        assert!(
            ego.mesh_brake_density > 0.2,
            "Lead brake should register: {}",
            ego.mesh_brake_density
        );
    }

    #[test]
    fn test_convoy_config() {
        let config = SwarmConfig::convoy_highway(5);
        assert_eq!(config.num_peers, 5);
        assert!((config.initial_spacing - 40.0).abs() < 0.01);
        assert!((config.target_speed - 13.4).abs() < 0.01);
    }

    #[test]
    fn test_body_frame_transform() {
        let config = SwarmConfig::convoy_straight(1);
        let swarm = SwarmSimulator::new(config);
        let ego = VehicleState::cruising(13.4); // heading=0, at origin
        let signals = swarm.mesh_signals(&ego);
        // Peer is ahead (positive x): should have positive longitudinal in body frame
        assert!(
            signals[0].relative_position[0] > 0.0,
            "Peer ahead should have positive lon: {}",
            signals[0].relative_position[0]
        );
        // Lateral should be near zero (same lane)
        assert!(
            signals[0].relative_position[1].abs() < 5.0,
            "Same lane → small lateral: {}",
            signals[0].relative_position[1]
        );
    }

    #[test]
    fn test_peer_relative_speed() {
        let config = SwarmConfig::convoy_straight(1);
        let swarm = SwarmSimulator::new(config);
        let mut ego = VehicleState::cruising(13.4);
        swarm.aggregate_mesh(&mut ego);
        // Both at same speed → relative speed near 0
        assert!(
            ego.peer_relative_speed.abs() < 1.0,
            "Same speed → small relative: {}",
            ego.peer_relative_speed
        );
    }

    #[test]
    fn test_road_following() {
        let config = SwarmConfig::convoy_highway(2);
        let mut swarm = SwarmSimulator::new(config);
        // Step peers for a while
        for _ in 0..500 {
            swarm.step(0.005);
        }
        // All peers should still be near the road (finite state)
        for peer in swarm.peers() {
            assert!(
                peer.sim.state().is_finite(),
                "Peer state should be finite: {:?}",
                peer.sim.state()
            );
        }
    }

    #[test]
    fn test_reset_braking() {
        let config = SwarmConfig::convoy_straight(3);
        let mut swarm = SwarmSimulator::new(config);
        swarm.trigger_lead_brake(0);
        swarm.trigger_lead_brake(2);
        assert!(swarm.peers()[0].is_braking);
        assert!(swarm.peers()[2].is_braking);

        swarm.reset_braking();
        for peer in swarm.peers() {
            assert!(
                !peer.is_braking,
                "All peers should stop braking after reset"
            );
        }
    }

    #[test]
    fn test_mesh_signal_fields() {
        let config = SwarmConfig::convoy_straight(1);
        let swarm = SwarmSimulator::new(config);
        let ego = VehicleState::cruising(13.4);
        let signals = swarm.mesh_signals(&ego);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].peer_id, 1);
        assert!(signals[0].peer_speed > 0.0);
        assert!(!signals[0].peer_braking);
        assert!(!signals[0].peer_abs_engaged);
        assert!((signals[0].latency - 0.01).abs() < 0.001);
    }
}
