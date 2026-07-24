// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-dimensional tunnel occupancy and right-of-way reservations.
//!
//! The reference plant models a single depth axis, so collision avoidance is
//! expressed as bounded depth intervals rather than pretending to know a full
//! 3-D pose. Reservations are ordered by epoch/sequence and resolved by
//! explicit emergency priority followed by stable agent-id tie breaking.

use crate::team::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_OCCUPANCY_CAPACITY: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TunnelDirection {
    Outbound,
    Inbound,
    Holding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReservationPriority {
    Routine,
    Return,
    Rescue,
    Emergency,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TunnelReservation {
    pub agent_id: AgentId,
    pub epoch: u32,
    pub sequence: u64,
    pub issued_step: u64,
    pub valid_from_step: u64,
    pub valid_until_step: u64,
    pub minimum_depth_m: f64,
    pub maximum_depth_m: f64,
    pub direction: TunnelDirection,
    pub priority: ReservationPriority,
}

impl TunnelReservation {
    pub fn is_valid(self) -> bool {
        self.agent_id != AgentId::SURFACE_CONTROL
            && self.valid_until_step >= self.valid_from_step
            && self.minimum_depth_m.is_finite()
            && self.maximum_depth_m.is_finite()
            && self.minimum_depth_m >= 0.0
            && self.maximum_depth_m >= self.minimum_depth_m
            && self.maximum_depth_m <= 200.0
    }

    pub fn active_at(self, step: u64) -> bool {
        step >= self.valid_from_step && step <= self.valid_until_step
    }

    fn version_is_newer_than(self, other: Self) -> bool {
        self.epoch > other.epoch || (self.epoch == other.epoch && self.sequence > other.sequence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationRejection {
    Invalid,
    SelfMessage,
    Replay,
    Capacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OccupancyAssessment {
    pub active_reservations: usize,
    pub conflicting_agent: Option<AgentId>,
    pub minimum_separation_m: f64,
    pub conflict_severity: f32,
    pub must_yield: bool,
    pub peer_priority: Option<ReservationPriority>,
}

impl OccupancyAssessment {
    pub const fn clear() -> Self {
        Self {
            active_reservations: 0,
            conflicting_agent: None,
            minimum_separation_m: f64::INFINITY,
            conflict_severity: 0.0,
            must_yield: false,
            peer_priority: None,
        }
    }

    pub fn conflict(self) -> bool {
        self.conflicting_agent.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunnelOccupancy {
    local_agent: AgentId,
    capacity: usize,
    reservations: BTreeMap<AgentId, TunnelReservation>,
}

impl TunnelOccupancy {
    pub fn new(local_agent: AgentId, capacity: usize) -> Self {
        Self {
            local_agent,
            capacity: capacity.max(1),
            reservations: BTreeMap::new(),
        }
    }

    pub fn ingest(&mut self, reservation: TunnelReservation) -> Result<(), ReservationRejection> {
        if !reservation.is_valid() {
            return Err(ReservationRejection::Invalid);
        }
        if reservation.agent_id == self.local_agent {
            return Err(ReservationRejection::SelfMessage);
        }
        if let Some(existing) = self.reservations.get(&reservation.agent_id).copied() {
            if !reservation.version_is_newer_than(existing) {
                return Err(ReservationRejection::Replay);
            }
        } else if self.reservations.len() >= self.capacity {
            return Err(ReservationRejection::Capacity);
        }
        self.reservations.insert(reservation.agent_id, reservation);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn assess(
        &self,
        current_step: u64,
        local_depth_m: f64,
        local_direction: TunnelDirection,
        local_priority: ReservationPriority,
        lookahead_m: f64,
        clearance_m: f64,
    ) -> OccupancyAssessment {
        if !local_depth_m.is_finite() {
            return OccupancyAssessment {
                active_reservations: 0,
                conflicting_agent: None,
                minimum_separation_m: 0.0,
                conflict_severity: 1.0,
                must_yield: true,
                peer_priority: None,
            };
        }
        let lookahead = if lookahead_m.is_finite() {
            lookahead_m.max(0.0)
        } else {
            0.0
        };
        let clearance = if clearance_m.is_finite() {
            clearance_m.max(0.0)
        } else {
            0.0
        };
        let (local_min, local_max) = match local_direction {
            TunnelDirection::Outbound => (local_depth_m, local_depth_m + lookahead),
            TunnelDirection::Inbound => ((local_depth_m - lookahead).max(0.0), local_depth_m),
            TunnelDirection::Holding => (
                (local_depth_m - clearance).max(0.0),
                local_depth_m + clearance,
            ),
        };
        let mut result = OccupancyAssessment::clear();
        for reservation in self
            .reservations
            .values()
            .copied()
            .filter(|reservation| reservation.active_at(current_step))
        {
            result.active_reservations += 1;
            let separated = reservation.maximum_depth_m + clearance < local_min
                || reservation.minimum_depth_m > local_max + clearance;
            let separation = if reservation.maximum_depth_m < local_min {
                local_min - reservation.maximum_depth_m
            } else if reservation.minimum_depth_m > local_max {
                reservation.minimum_depth_m - local_max
            } else {
                0.0
            };
            result.minimum_separation_m = result.minimum_separation_m.min(separation);
            let opposed = matches!(local_direction, TunnelDirection::Holding)
                || matches!(reservation.direction, TunnelDirection::Holding)
                || local_direction != reservation.direction;
            if separated || !opposed {
                continue;
            }
            let peer_wins = reservation.priority > local_priority
                || (reservation.priority == local_priority
                    && reservation.agent_id < self.local_agent);
            let severity = if separation <= clearance * 0.25 {
                1.0
            } else if clearance <= f64::EPSILON {
                0.9
            } else {
                (1.0 - separation / clearance).clamp(0.55, 1.0) as f32
            };
            let replace = result.conflicting_agent.is_none()
                || severity > result.conflict_severity
                || (severity == result.conflict_severity && peer_wins && !result.must_yield);
            if replace {
                result.conflicting_agent = Some(reservation.agent_id);
                result.conflict_severity = severity;
                result.must_yield = peer_wins;
                result.peer_priority = Some(reservation.priority);
            }
        }
        result
    }

    pub fn active_reservations(
        &self,
        current_step: u64,
    ) -> impl Iterator<Item = TunnelReservation> + '_ {
        self.reservations
            .values()
            .copied()
            .filter(move |reservation| reservation.active_at(current_step))
    }

    pub fn clear(&mut self) {
        self.reservations.clear();
    }
}

impl Default for TunnelOccupancy {
    fn default() -> Self {
        Self::new(AgentId::new(1), DEFAULT_OCCUPANCY_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reservation(agent: u64, direction: TunnelDirection) -> TunnelReservation {
        TunnelReservation {
            agent_id: AgentId::new(agent),
            epoch: 1,
            sequence: 1,
            issued_step: 0,
            valid_from_step: 0,
            valid_until_step: 100,
            minimum_depth_m: 9.0,
            maximum_depth_m: 11.0,
            direction,
            priority: ReservationPriority::Routine,
        }
    }

    #[test]
    fn opposing_reservations_create_a_conflict() {
        let mut occupancy = TunnelOccupancy::new(AgentId::new(3), 4);
        assert_eq!(
            occupancy.ingest(reservation(2, TunnelDirection::Inbound)),
            Ok(())
        );
        let assessment = occupancy.assess(
            10,
            8.0,
            TunnelDirection::Outbound,
            ReservationPriority::Routine,
            4.0,
            1.0,
        );
        assert!(assessment.conflict());
        assert!(assessment.must_yield);
        assert_eq!(assessment.conflicting_agent, Some(AgentId::new(2)));
    }

    #[test]
    fn same_direction_convoy_does_not_force_yield() {
        let mut occupancy = TunnelOccupancy::new(AgentId::new(3), 4);
        assert_eq!(
            occupancy.ingest(reservation(2, TunnelDirection::Outbound)),
            Ok(())
        );
        let assessment = occupancy.assess(
            10,
            8.0,
            TunnelDirection::Outbound,
            ReservationPriority::Routine,
            4.0,
            1.0,
        );
        assert!(!assessment.conflict());
    }

    #[test]
    fn emergency_priority_wins_even_with_higher_agent_id() {
        let mut occupancy = TunnelOccupancy::new(AgentId::new(1), 4);
        let mut peer = reservation(9, TunnelDirection::Inbound);
        peer.priority = ReservationPriority::Emergency;
        assert_eq!(occupancy.ingest(peer), Ok(()));
        let assessment = occupancy.assess(
            10,
            8.0,
            TunnelDirection::Outbound,
            ReservationPriority::Routine,
            4.0,
            1.0,
        );
        assert!(assessment.must_yield);
    }
}
