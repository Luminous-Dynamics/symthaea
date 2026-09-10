// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fault campaign for the optional EVA rescue-propulsion subsystem.
//!
//! This campaign is separate from PLSS survival faults because rescue
//! propulsion is intentionally non-survival-critical. Losing rescue propulsion
//! must never remove breathing, pressure, or thermal-control authority.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RescueFault {
    PropellantLeak,
    IsolationValveStuckClosed,
    IsolationValveStuckOpen,
    RescuePowerUnavailable,
    IndependentImuInvalid,
    RelativeNavigationInvalid,
    ThrusterStuckOff,
    ThrusterStuckOn,
    WatchdogFailure,
    NoFireGeometryInvalid,
}

pub const ALL_RESCUE_FAULTS: [RescueFault; 10] = [
    RescueFault::PropellantLeak,
    RescueFault::IsolationValveStuckClosed,
    RescueFault::IsolationValveStuckOpen,
    RescueFault::RescuePowerUnavailable,
    RescueFault::IndependentImuInvalid,
    RescueFault::RelativeNavigationInvalid,
    RescueFault::ThrusterStuckOff,
    RescueFault::ThrusterStuckOn,
    RescueFault::WatchdogFailure,
    RescueFault::NoFireGeometryInvalid,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RescueFaultDisposition {
    ContinueWithoutAutomaticRescue,
    DepowerAndIsolatePropulsion,
    ReturnToSafeHaven,
    ImmediateIsolation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueFaultScenario {
    pub faults: Vec<RescueFault>,
}

impl RescueFaultScenario {
    pub fn single(fault: RescueFault) -> Self {
        Self { faults: vec![fault] }
    }

    pub fn pair(a: RescueFault, b: RescueFault) -> Self {
        if a == b {
            Self { faults: vec![a] }
        } else {
            Self { faults: vec![a, b] }
        }
    }

    pub fn disposition(&self) -> RescueFaultDisposition {
        // Uncommanded thrust or inability to close/isolate a flow path is the
        // highest-priority propulsion-specific failure. This does not command
        // PLSS; it demands independent propulsion isolation.
        if self.faults.contains(&RescueFault::ThrusterStuckOn)
            || self.faults.contains(&RescueFault::IsolationValveStuckOpen)
        {
            return RescueFaultDisposition::ImmediateIsolation;
        }

        if self.faults.contains(&RescueFault::PropellantLeak)
            || self.faults.contains(&RescueFault::NoFireGeometryInvalid)
        {
            return RescueFaultDisposition::ReturnToSafeHaven;
        }

        if self.faults.contains(&RescueFault::WatchdogFailure)
            || self.faults.contains(&RescueFault::RescuePowerUnavailable)
            || self.faults.contains(&RescueFault::IsolationValveStuckClosed)
            || self.faults.contains(&RescueFault::IndependentImuInvalid)
            || self.faults.contains(&RescueFault::ThrusterStuckOff)
        {
            return RescueFaultDisposition::DepowerAndIsolatePropulsion;
        }

        if self.faults.contains(&RescueFault::RelativeNavigationInvalid) {
            return RescueFaultDisposition::ContinueWithoutAutomaticRescue;
        }

        RescueFaultDisposition::ContinueWithoutAutomaticRescue
    }
}

pub fn single_fault_campaign() -> Vec<RescueFaultScenario> {
    ALL_RESCUE_FAULTS
        .into_iter()
        .map(RescueFaultScenario::single)
        .collect()
}

pub fn pairwise_fault_campaign() -> Vec<RescueFaultScenario> {
    let mut cases = Vec::new();
    for (i, a) in ALL_RESCUE_FAULTS.iter().copied().enumerate() {
        for b in ALL_RESCUE_FAULTS.iter().copied().skip(i + 1) {
            cases.push(RescueFaultScenario::pair(a, b));
        }
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rescue_fault_has_a_deterministic_disposition() {
        for scenario in single_fault_campaign() {
            let _ = scenario.disposition();
        }
    }

    #[test]
    fn pairwise_campaign_is_complete() {
        let n = ALL_RESCUE_FAULTS.len();
        assert_eq!(pairwise_fault_campaign().len(), n * (n - 1) / 2);
    }

    #[test]
    fn stuck_on_thruster_demands_immediate_isolation() {
        let s = RescueFaultScenario::single(RescueFault::ThrusterStuckOn);
        assert_eq!(s.disposition(), RescueFaultDisposition::ImmediateIsolation);
    }

    #[test]
    fn invalid_relative_navigation_removes_automatic_rescue_not_plss() {
        let s = RescueFaultScenario::single(RescueFault::RelativeNavigationInvalid);
        assert_eq!(
            s.disposition(),
            RescueFaultDisposition::ContinueWithoutAutomaticRescue
        );
    }

    #[test]
    fn dangerous_fault_dominates_navigation_loss() {
        let s = RescueFaultScenario::pair(
            RescueFault::RelativeNavigationInvalid,
            RescueFault::IsolationValveStuckOpen,
        );
        assert_eq!(s.disposition(), RescueFaultDisposition::ImmediateIsolation);
    }
}
