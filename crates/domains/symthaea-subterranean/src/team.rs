// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded peer directory and replay-resistant team status exchange.
//!
//! This module deliberately does not claim cryptographic authenticity. It
//! provides deterministic identity, epoch, sequence, freshness, and range
//! checks so a transport can authenticate messages externally without leaving
//! ordering or stale-peer semantics ambiguous inside the platform.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_TEAM_CAPACITY: usize = 16;
pub const DEFAULT_PEER_STALE_STEPS: u64 = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentId(pub u64);

impl AgentId {
    pub const SURFACE_CONTROL: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeamRole {
    Scout,
    Borer,
    Mapper,
    Relay,
    Rescue,
    SurfaceControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerCondition {
    Nominal,
    Degraded,
    Holding,
    Withdrawing,
    Distress,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeamHeartbeat {
    pub agent_id: AgentId,
    pub epoch: u32,
    pub sequence: u64,
    pub emitted_step: u64,
    pub role: TeamRole,
    pub condition: PeerCondition,
    pub depth_m: f64,
    pub battery_ratio: f64,
    pub route_confidence: f64,
    pub link_quality: f64,
    pub hazard_severity: f32,
}

impl TeamHeartbeat {
    pub fn is_valid(self) -> bool {
        self.agent_id != AgentId::SURFACE_CONTROL
            && self.depth_m.is_finite()
            && (0.0..=200.0).contains(&self.depth_m)
            && self.battery_ratio.is_finite()
            && (0.0..=1.0).contains(&self.battery_ratio)
            && self.route_confidence.is_finite()
            && (0.0..=1.0).contains(&self.route_confidence)
            && self.link_quality.is_finite()
            && (0.0..=1.0).contains(&self.link_quality)
            && self.hazard_severity.is_finite()
            && (0.0..=1.0).contains(&self.hazard_severity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeartbeatRejection {
    Invalid,
    SelfMessage,
    Replay,
    Capacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PeerRecord {
    pub heartbeat: TeamHeartbeat,
    pub received_step: u64,
}

impl PeerRecord {
    pub fn age_steps(self, current_step: u64) -> u64 {
        current_step.saturating_sub(self.received_step)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TeamStatus {
    pub known_peers: usize,
    pub fresh_peers: usize,
    pub stale_peers: usize,
    pub distressed_peers: usize,
    pub minimum_link_quality: f64,
    pub minimum_battery_ratio: f64,
    pub maximum_hazard_severity: f32,
}

impl TeamStatus {
    pub const fn alone() -> Self {
        Self {
            known_peers: 0,
            fresh_peers: 0,
            stale_peers: 0,
            distressed_peers: 0,
            minimum_link_quality: 1.0,
            minimum_battery_ratio: 1.0,
            maximum_hazard_severity: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeamDirectory {
    local_agent: AgentId,
    capacity: usize,
    stale_after_steps: u64,
    peers: BTreeMap<AgentId, PeerRecord>,
}

impl TeamDirectory {
    pub fn new(local_agent: AgentId, capacity: usize, stale_after_steps: u64) -> Self {
        Self {
            local_agent,
            capacity: capacity.max(1),
            stale_after_steps: stale_after_steps.max(1),
            peers: BTreeMap::new(),
        }
    }

    pub fn ingest(
        &mut self,
        heartbeat: TeamHeartbeat,
        received_step: u64,
    ) -> Result<(), HeartbeatRejection> {
        if !heartbeat.is_valid() {
            return Err(HeartbeatRejection::Invalid);
        }
        if heartbeat.agent_id == self.local_agent {
            return Err(HeartbeatRejection::SelfMessage);
        }
        if let Some(existing) = self.peers.get(&heartbeat.agent_id) {
            let old = existing.heartbeat;
            let ordered = heartbeat.epoch > old.epoch
                || (heartbeat.epoch == old.epoch && heartbeat.sequence > old.sequence);
            if !ordered {
                return Err(HeartbeatRejection::Replay);
            }
        } else if self.peers.len() >= self.capacity {
            return Err(HeartbeatRejection::Capacity);
        }
        self.peers.insert(
            heartbeat.agent_id,
            PeerRecord {
                heartbeat,
                received_step,
            },
        );
        Ok(())
    }

    pub fn peer(&self, agent_id: AgentId) -> Option<PeerRecord> {
        self.peers.get(&agent_id).copied()
    }

    pub fn freshest_distress(&self, current_step: u64) -> Option<PeerRecord> {
        self.peers
            .values()
            .copied()
            .filter(|record| record.age_steps(current_step) <= self.stale_after_steps)
            .filter(|record| {
                matches!(
                    record.heartbeat.condition,
                    PeerCondition::Distress | PeerCondition::Lost
                )
            })
            .max_by(|left, right| {
                left.heartbeat
                    .hazard_severity
                    .total_cmp(&right.heartbeat.hazard_severity)
                    .then_with(|| right.received_step.cmp(&left.received_step))
                    .then_with(|| right.heartbeat.agent_id.cmp(&left.heartbeat.agent_id))
            })
    }

    pub fn status(&self, current_step: u64) -> TeamStatus {
        if self.peers.is_empty() {
            return TeamStatus::alone();
        }
        let mut status = TeamStatus {
            known_peers: self.peers.len(),
            fresh_peers: 0,
            stale_peers: 0,
            distressed_peers: 0,
            minimum_link_quality: 1.0,
            minimum_battery_ratio: 1.0,
            maximum_hazard_severity: 0.0,
        };
        for record in self.peers.values().copied() {
            if record.age_steps(current_step) > self.stale_after_steps {
                status.stale_peers += 1;
                continue;
            }
            status.fresh_peers += 1;
            status.minimum_link_quality = status
                .minimum_link_quality
                .min(record.heartbeat.link_quality);
            status.minimum_battery_ratio = status
                .minimum_battery_ratio
                .min(record.heartbeat.battery_ratio);
            status.maximum_hazard_severity = status
                .maximum_hazard_severity
                .max(record.heartbeat.hazard_severity);
            if matches!(
                record.heartbeat.condition,
                PeerCondition::Distress | PeerCondition::Lost
            ) {
                status.distressed_peers += 1;
            }
        }
        status
    }

    pub fn local_agent(&self) -> AgentId {
        self.local_agent
    }

    pub fn peers(&self) -> impl Iterator<Item = PeerRecord> + '_ {
        self.peers.values().copied()
    }

    pub fn clear(&mut self) {
        self.peers.clear();
    }
}

impl Default for TeamDirectory {
    fn default() -> Self {
        Self::new(
            AgentId::new(1),
            DEFAULT_TEAM_CAPACITY,
            DEFAULT_PEER_STALE_STEPS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heartbeat(agent: u64, sequence: u64) -> TeamHeartbeat {
        TeamHeartbeat {
            agent_id: AgentId::new(agent),
            epoch: 1,
            sequence,
            emitted_step: sequence,
            role: TeamRole::Scout,
            condition: PeerCondition::Nominal,
            depth_m: 12.0,
            battery_ratio: 0.8,
            route_confidence: 0.9,
            link_quality: 0.7,
            hazard_severity: 0.0,
        }
    }

    #[test]
    fn rejects_replay_and_self_messages() {
        let mut directory = TeamDirectory::new(AgentId::new(7), 4, 10);
        assert_eq!(
            directory.ingest(heartbeat(7, 1), 1),
            Err(HeartbeatRejection::SelfMessage)
        );
        assert_eq!(directory.ingest(heartbeat(8, 1), 1), Ok(()));
        assert_eq!(
            directory.ingest(heartbeat(8, 1), 2),
            Err(HeartbeatRejection::Replay)
        );
        assert_eq!(directory.ingest(heartbeat(8, 2), 2), Ok(()));
    }

    #[test]
    fn stale_peers_are_not_reported_as_fresh_distress() {
        let mut directory = TeamDirectory::new(AgentId::new(1), 4, 10);
        let mut distressed = heartbeat(2, 1);
        distressed.condition = PeerCondition::Distress;
        distressed.hazard_severity = 0.9;
        assert_eq!(directory.ingest(distressed, 5), Ok(()));
        assert!(directory.freshest_distress(10).is_some());
        assert!(directory.freshest_distress(16).is_none());
        assert_eq!(directory.status(16).stale_peers, 1);
    }

    #[test]
    fn capacity_is_bounded_without_implicit_eviction() {
        let mut directory = TeamDirectory::new(AgentId::new(1), 1, 10);
        assert_eq!(directory.ingest(heartbeat(2, 1), 1), Ok(()));
        assert_eq!(
            directory.ingest(heartbeat(3, 1), 1),
            Err(HeartbeatRejection::Capacity)
        );
    }
}
