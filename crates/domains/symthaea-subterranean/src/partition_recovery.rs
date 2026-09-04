// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic communication-partition survival and bounded reconciliation.
//!
//! A restored radio link is not equivalent to restored operational truth.
//! After a partition, the platform holds motion while fresh team state is
//! observed for a bounded reconciliation dwell. This module does not perform
//! cryptographic peer authentication; that remains a transport responsibility.

use crate::mission::SubterraneanMissionIntent;
use crate::types::SubterraneanCommand;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionRecoveryMode {
    Connected,
    Grace,
    LocalAutonomy,
    ReturnToMesh,
    HoldAndBeacon,
    Reconciling,
}

impl PartitionRecoveryMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Connected => "connected",
            Self::Grace => "grace",
            Self::LocalAutonomy => "local_autonomy",
            Self::ReturnToMesh => "return_to_mesh",
            Self::HoldAndBeacon => "hold_and_beacon",
            Self::Reconciling => "reconciling",
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::Connected | Self::Grace | Self::LocalAutonomy => None,
            Self::ReturnToMesh => Some(SubterraneanMissionIntent::ReturnHome),
            Self::HoldAndBeacon | Self::Reconciling => {
                Some(SubterraneanMissionIntent::MaintainRelay)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionRecoveryPolicy {
    pub grace_steps: u32,
    pub local_autonomy_steps: u32,
    pub reconciliation_dwell_steps: u32,
    pub minimum_battery_for_local_autonomy: f64,
}

impl Default for PartitionRecoveryPolicy {
    fn default() -> Self {
        Self {
            grace_steps: 200,
            local_autonomy_steps: 2_000,
            reconciliation_dwell_steps: 20,
            minimum_battery_for_local_autonomy: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionObservation {
    pub surface_reachable: bool,
    pub fresh_peers: usize,
    pub battery_ratio: f64,
    pub return_feasible: bool,
    pub local_map_revision: u64,
    pub highest_peer_map_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionRecoveryAssessment {
    pub mode: PartitionRecoveryMode,
    pub partition_steps: u32,
    pub reconciliation_steps: u32,
    pub map_revision_gap: u64,
    pub motion_permitted: bool,
    pub team_state_authoritative: bool,
}

impl PartitionRecoveryAssessment {
    pub const fn connected() -> Self {
        Self {
            mode: PartitionRecoveryMode::Connected,
            partition_steps: 0,
            reconciliation_steps: 0,
            map_revision_gap: 0,
            motion_permitted: true,
            team_state_authoritative: true,
        }
    }

    pub fn constrain_nominal(self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        if !self.motion_permitted {
            command.set_cutter_head(0.0);
            command.set_auger_feed(0.0);
            command.set_left_track(0.0);
            command.set_right_track(0.0);
            command.set_ballast_trim(0.0);
        }
        command.sanitize();
        command
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionRecoverySupervisor {
    policy: PartitionRecoveryPolicy,
    mode: PartitionRecoveryMode,
    partition_steps: u32,
    reconciliation_steps: u32,
    transitions: u64,
    partitions: u64,
    reconciliations: u64,
    last_assessment: PartitionRecoveryAssessment,
}

impl PartitionRecoverySupervisor {
    pub fn new(policy: PartitionRecoveryPolicy) -> Self {
        Self {
            policy,
            mode: PartitionRecoveryMode::Connected,
            partition_steps: 0,
            reconciliation_steps: 0,
            transitions: 0,
            partitions: 0,
            reconciliations: 0,
            last_assessment: PartitionRecoveryAssessment::connected(),
        }
    }

    pub fn validate(&self) -> bool {
        self.policy.reconciliation_dwell_steps > 0
            && self.policy.minimum_battery_for_local_autonomy.is_finite()
            && (0.0..=1.0).contains(&self.policy.minimum_battery_for_local_autonomy)
            && self.last_assessment.map_revision_gap
                == self.last_assessment.map_revision_gap.min(u64::MAX)
    }

    pub fn update(&mut self, observation: PartitionObservation) -> PartitionRecoveryAssessment {
        let previous = self.mode;
        let connected = observation.surface_reachable;
        if !connected {
            if previous == PartitionRecoveryMode::Connected {
                self.partitions = self.partitions.saturating_add(1);
            }
            self.partition_steps = self.partition_steps.saturating_add(1);
            self.reconciliation_steps = 0;
            self.mode = if self.partition_steps <= self.policy.grace_steps {
                PartitionRecoveryMode::Grace
            } else if self.partition_steps <= self.policy.local_autonomy_steps
                && observation.battery_ratio.is_finite()
                && observation.battery_ratio >= self.policy.minimum_battery_for_local_autonomy
            {
                PartitionRecoveryMode::LocalAutonomy
            } else if observation.return_feasible {
                PartitionRecoveryMode::ReturnToMesh
            } else {
                PartitionRecoveryMode::HoldAndBeacon
            };
        } else if self.partition_steps > 0
            || matches!(
                previous,
                PartitionRecoveryMode::Reconciling
                    | PartitionRecoveryMode::ReturnToMesh
                    | PartitionRecoveryMode::HoldAndBeacon
                    | PartitionRecoveryMode::LocalAutonomy
                    | PartitionRecoveryMode::Grace
            )
        {
            self.mode = PartitionRecoveryMode::Reconciling;
            let revisions_converged = observation.local_map_revision
                == observation.highest_peer_map_revision
                || observation.fresh_peers == 0;
            if revisions_converged {
                self.reconciliation_steps = self.reconciliation_steps.saturating_add(1);
            } else {
                self.reconciliation_steps = 0;
            }
            if self.reconciliation_steps >= self.policy.reconciliation_dwell_steps {
                self.mode = PartitionRecoveryMode::Connected;
                self.partition_steps = 0;
                self.reconciliation_steps = 0;
                self.reconciliations = self.reconciliations.saturating_add(1);
            }
        } else {
            self.mode = PartitionRecoveryMode::Connected;
            self.partition_steps = 0;
            self.reconciliation_steps = 0;
        }

        if self.mode != previous {
            self.transitions = self.transitions.saturating_add(1);
        }
        let map_revision_gap = observation
            .local_map_revision
            .abs_diff(observation.highest_peer_map_revision);
        self.last_assessment = PartitionRecoveryAssessment {
            mode: self.mode,
            partition_steps: self.partition_steps,
            reconciliation_steps: self.reconciliation_steps,
            map_revision_gap,
            motion_permitted: !matches!(
                self.mode,
                PartitionRecoveryMode::HoldAndBeacon | PartitionRecoveryMode::Reconciling
            ),
            team_state_authoritative: self.mode == PartitionRecoveryMode::Connected,
        };
        self.last_assessment
    }

    pub const fn assessment(&self) -> PartitionRecoveryAssessment {
        self.last_assessment
    }

    pub const fn transitions(&self) -> u64 {
        self.transitions
    }

    pub const fn partitions(&self) -> u64 {
        self.partitions
    }

    pub const fn reconciliations(&self) -> u64 {
        self.reconciliations
    }

    /// A runtime restart breaks continuity of team-state authority even when the
    /// transport currently appears reachable. Enter reconciliation without
    /// fabricating a network-partition event, preserve historical partition and
    /// map-gap evidence, and discard only positive reconciliation dwell credit.
    pub(crate) fn enter_operational_restart_reconciliation(&mut self) {
        let previous = self.mode;
        self.mode = PartitionRecoveryMode::Reconciling;
        self.reconciliation_steps = 0;
        if self.mode != previous {
            self.transitions = self.transitions.saturating_add(1);
        }
        self.last_assessment = PartitionRecoveryAssessment {
            mode: PartitionRecoveryMode::Reconciling,
            partition_steps: self.partition_steps,
            reconciliation_steps: 0,
            map_revision_gap: self.last_assessment.map_revision_gap,
            motion_permitted: false,
            team_state_authoritative: false,
        };
    }

    pub fn reset_runtime(&mut self) {
        self.mode = PartitionRecoveryMode::Connected;
        self.partition_steps = 0;
        self.reconciliation_steps = 0;
        self.last_assessment = PartitionRecoveryAssessment::connected();
    }
}

impl Default for PartitionRecoverySupervisor {
    fn default() -> Self {
        Self::new(PartitionRecoveryPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(surface_reachable: bool) -> PartitionObservation {
        PartitionObservation {
            surface_reachable,
            fresh_peers: 1,
            battery_ratio: 0.8,
            return_feasible: true,
            local_map_revision: 4,
            highest_peer_map_revision: 4,
        }
    }

    #[test]
    fn prolonged_partition_returns_to_mesh() {
        let mut supervisor = PartitionRecoverySupervisor::new(PartitionRecoveryPolicy {
            grace_steps: 1,
            local_autonomy_steps: 2,
            ..Default::default()
        });
        supervisor.update(observation(false));
        supervisor.update(observation(false));
        let assessment = supervisor.update(observation(false));
        assert_eq!(assessment.mode, PartitionRecoveryMode::ReturnToMesh);
        assert_eq!(
            assessment.mode.mission_override(),
            Some(SubterraneanMissionIntent::ReturnHome)
        );
    }

    #[test]
    fn reconnection_holds_until_revision_dwell() {
        let mut supervisor = PartitionRecoverySupervisor::new(PartitionRecoveryPolicy {
            grace_steps: 0,
            local_autonomy_steps: 1,
            reconciliation_dwell_steps: 2,
            ..Default::default()
        });
        supervisor.update(observation(false));
        let first = supervisor.update(observation(true));
        assert_eq!(first.mode, PartitionRecoveryMode::Reconciling);
        assert!(!first.motion_permitted);
        let second = supervisor.update(observation(true));
        assert_eq!(second.mode, PartitionRecoveryMode::Connected);
    }

    #[test]
    fn revision_gap_resets_reconciliation_dwell() {
        let mut supervisor = PartitionRecoverySupervisor::new(PartitionRecoveryPolicy {
            grace_steps: 0,
            local_autonomy_steps: 1,
            reconciliation_dwell_steps: 2,
            ..Default::default()
        });
        supervisor.update(observation(false));
        let mut diverged = observation(true);
        diverged.highest_peer_map_revision = 9;
        let assessment = supervisor.update(diverged);
        assert_eq!(assessment.reconciliation_steps, 0);
        assert_eq!(assessment.map_revision_gap, 5);
    }

    #[test]
    fn operational_restart_forces_reconciliation_without_counting_partition() {
        let mut supervisor = PartitionRecoverySupervisor::default();
        let partitions_before = supervisor.partitions();
        supervisor.enter_operational_restart_reconciliation();
        let assessment = supervisor.assessment();
        assert_eq!(assessment.mode, PartitionRecoveryMode::Reconciling);
        assert!(!assessment.motion_permitted);
        assert!(!assessment.team_state_authoritative);
        assert_eq!(assessment.reconciliation_steps, 0);
        assert_eq!(supervisor.partitions(), partitions_before);
    }
}
