// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composed field capability assessment for graceful mission degradation.

use crate::actuator_isolation::{ActuatorIsolationReport, PhysicalActuator};
use crate::field_envelope::{FieldEnvelopeAssessment, FieldEnvelopeMode};
use crate::maintenance::MaintenanceAssessment;
use crate::mission::SubterraneanMissionIntent;
use crate::sensor_redundancy::SensorFusionReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityDisposition {
    FullMission,
    ReducedWork,
    ReturnOnly,
    HoldForRecovery,
}

impl CapabilityDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::FullMission => "full_mission",
            Self::ReducedWork => "reduced_work",
            Self::ReturnOnly => "return_only",
            Self::HoldForRecovery => "hold_for_recovery",
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::FullMission | Self::ReducedWork => None,
            Self::ReturnOnly => Some(SubterraneanMissionIntent::ReturnHome),
            Self::HoldForRecovery => Some(SubterraneanMissionIntent::HoldPosition),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityProfile {
    pub disposition: CapabilityDisposition,
    pub sensor_quorum: bool,
    pub mobility_axes_available: u8,
    pub cutting_available: bool,
    pub spoil_handling_available: bool,
    pub cooling_available: bool,
    pub dewatering_available: bool,
    pub communications_available: bool,
    pub mission_work_allowed: bool,
    pub isolated_actuators: usize,
}

impl CapabilityProfile {
    pub const fn nominal() -> Self {
        Self {
            disposition: CapabilityDisposition::FullMission,
            sensor_quorum: true,
            mobility_axes_available: 2,
            cutting_available: true,
            spoil_handling_available: true,
            cooling_available: true,
            dewatering_available: true,
            communications_available: true,
            mission_work_allowed: true,
            isolated_actuators: 0,
        }
    }

    pub fn assess(
        sensor: SensorFusionReport,
        actuators: ActuatorIsolationReport,
        envelope: FieldEnvelopeAssessment,
        maintenance: MaintenanceAssessment,
    ) -> Self {
        let left_available = !actuators.is_isolated(PhysicalActuator::LeftTrack);
        let right_available = !actuators.is_isolated(PhysicalActuator::RightTrack);
        let mobility_axes_available = left_available as u8 + right_available as u8;
        let sensor_quorum = !sensor.requires_fail_closed();
        let cutting_available =
            !actuators.is_isolated(PhysicalActuator::Cutter) && envelope.cutter_cap > 0.0;
        let spoil_handling_available =
            !actuators.is_isolated(PhysicalActuator::Auger) && envelope.auger_cap > 0.0;
        let cooling_available =
            !actuators.is_isolated(PhysicalActuator::ThermalPump) && maintenance.cooling_available;
        let dewatering_available = !actuators.is_isolated(PhysicalActuator::DewateringPump);
        let communications_available = !actuators.is_isolated(PhysicalActuator::RelayDeployer);
        let mission_work_allowed = sensor_quorum
            && cutting_available
            && spoil_handling_available
            && envelope.mission_work_allowed
            && !maintenance.mission_abort_required;

        let disposition = if !sensor_quorum
            || mobility_axes_available == 0
            || !cooling_available
            || matches!(envelope.mode, FieldEnvelopeMode::SurvivalHold)
        {
            CapabilityDisposition::HoldForRecovery
        } else if mobility_axes_available < 2
            || matches!(
                envelope.mode,
                FieldEnvelopeMode::CriticalPower | FieldEnvelopeMode::ThermalProtection
            )
            || maintenance.mission_abort_required
        {
            CapabilityDisposition::ReturnOnly
        } else if !mission_work_allowed
            || actuators.isolated_count > 0
            || maintenance.maintenance_due
            || matches!(envelope.mode, FieldEnvelopeMode::Derated)
        {
            CapabilityDisposition::ReducedWork
        } else {
            CapabilityDisposition::FullMission
        };

        Self {
            disposition,
            sensor_quorum,
            mobility_axes_available,
            cutting_available,
            spoil_handling_available,
            cooling_available,
            dewatering_available,
            communications_available,
            mission_work_allowed,
            isolated_actuators: actuators.isolated_count,
        }
    }
}

impl Default for CapabilityProfile {
    fn default() -> Self {
        Self::nominal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_track_loss_degrades_to_return_only() {
        let mut actuators = ActuatorIsolationReport::nominal();
        actuators.isolated[PhysicalActuator::LeftTrack.index()] = true;
        actuators.isolated_count = 1;
        let profile = CapabilityProfile::assess(
            SensorFusionReport::nominal(),
            actuators,
            FieldEnvelopeAssessment::nominal(),
            MaintenanceAssessment::nominal(),
        );
        assert_eq!(profile.disposition, CapabilityDisposition::ReturnOnly);
        assert_eq!(profile.mobility_axes_available, 1);
    }

    #[test]
    fn critical_sensor_quorum_loss_forces_hold() {
        let mut sensor = SensorFusionReport::nominal();
        sensor.critical_channels_without_quorum = 1;
        let profile = CapabilityProfile::assess(
            sensor,
            ActuatorIsolationReport::nominal(),
            FieldEnvelopeAssessment::nominal(),
            MaintenanceAssessment::nominal(),
        );
        assert_eq!(profile.disposition, CapabilityDisposition::HoldForRecovery);
    }

    #[test]
    fn ordinary_derating_preserves_reduced_work_not_full_abort() {
        let mut envelope = FieldEnvelopeAssessment::nominal();
        envelope.mode = FieldEnvelopeMode::Derated;
        envelope.mission_work_allowed = false;
        let profile = CapabilityProfile::assess(
            SensorFusionReport::nominal(),
            ActuatorIsolationReport::nominal(),
            envelope,
            MaintenanceAssessment::nominal(),
        );
        assert_eq!(profile.disposition, CapabilityDisposition::ReducedWork);
    }
}
