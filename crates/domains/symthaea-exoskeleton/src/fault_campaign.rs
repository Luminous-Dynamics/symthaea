// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic EVA fault-campaign model for SX-010.
//!
//! The campaign defines expected degraded dispositions for single and combined
//! failures. It is a verification harness, not a substitute for hardware fault
//! trees, FMEA, or human-rating analysis.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaFault {
    PrimaryOxygenRegulatorFailure,
    SecondaryOxygenRegulatorFailure,
    ScrubberDegradation,
    PrimaryCoolingPumpFailure,
    SecondaryCoolingPathFailure,
    AssistMotorShort,
    SurvivalBatteryFailure,
    MobilityBatteryFailure,
    SuitOvertemperature,
    SensorDisagreement,
    DustDegradation,
    CommunicationLoss,
    RadiationAlert,
}

pub const ALL_EVA_FAULTS: [EvaFault; 13] = [
    EvaFault::PrimaryOxygenRegulatorFailure,
    EvaFault::SecondaryOxygenRegulatorFailure,
    EvaFault::ScrubberDegradation,
    EvaFault::PrimaryCoolingPumpFailure,
    EvaFault::SecondaryCoolingPathFailure,
    EvaFault::AssistMotorShort,
    EvaFault::SurvivalBatteryFailure,
    EvaFault::MobilityBatteryFailure,
    EvaFault::SuitOvertemperature,
    EvaFault::SensorDisagreement,
    EvaFault::DustDegradation,
    EvaFault::CommunicationLoss,
    EvaFault::RadiationAlert,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvaDisposition {
    ContinueNominal,
    ContinueLocalAutonomy,
    DegradeMission,
    DepowerAssist,
    ReturnToSafeHaven,
    AbortEva,
    ImmediateShelter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultScenario {
    pub faults: Vec<EvaFault>,
}

impl FaultScenario {
    pub fn single(fault: EvaFault) -> Self {
        Self { faults: vec![fault] }
    }

    pub fn pair(a: EvaFault, b: EvaFault) -> Self {
        if a == b {
            Self { faults: vec![a] }
        } else {
            Self { faults: vec![a, b] }
        }
    }

    pub fn disposition(&self) -> EvaDisposition {
        // Radiation alert demands the fastest path to materially better
        // shielding and dominates ordinary equipment-degradation actions.
        if self.faults.contains(&EvaFault::RadiationAlert) {
            return EvaDisposition::ImmediateShelter;
        }

        let both_oxygen_paths_lost = self
            .faults
            .contains(&EvaFault::PrimaryOxygenRegulatorFailure)
            && self
                .faults
                .contains(&EvaFault::SecondaryOxygenRegulatorFailure);
        let both_thermal_paths_lost = self
            .faults
            .contains(&EvaFault::PrimaryCoolingPumpFailure)
            && self
                .faults
                .contains(&EvaFault::SecondaryCoolingPathFailure);
        if both_oxygen_paths_lost
            || both_thermal_paths_lost
            || self.faults.contains(&EvaFault::SurvivalBatteryFailure)
            || self.faults.contains(&EvaFault::SuitOvertemperature)
        {
            return EvaDisposition::AbortEva;
        }

        if self.faults.contains(&EvaFault::ScrubberDegradation)
            || self
                .faults
                .contains(&EvaFault::PrimaryOxygenRegulatorFailure)
            || self.faults.contains(&EvaFault::PrimaryCoolingPumpFailure)
        {
            return EvaDisposition::ReturnToSafeHaven;
        }

        if self.faults.contains(&EvaFault::AssistMotorShort)
            || self.faults.contains(&EvaFault::MobilityBatteryFailure)
            || self.faults.contains(&EvaFault::SensorDisagreement)
        {
            return EvaDisposition::DepowerAssist;
        }

        if self.faults.contains(&EvaFault::DustDegradation) {
            return EvaDisposition::DegradeMission;
        }

        if self.faults.contains(&EvaFault::CommunicationLoss) {
            return EvaDisposition::ContinueLocalAutonomy;
        }

        EvaDisposition::ContinueNominal
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultCampaignCase {
    pub scenario: FaultScenario,
    pub expected: EvaDisposition,
}

pub fn single_fault_campaign() -> Vec<FaultCampaignCase> {
    ALL_EVA_FAULTS
        .into_iter()
        .map(|fault| {
            let scenario = FaultScenario::single(fault);
            let expected = scenario.disposition();
            FaultCampaignCase { scenario, expected }
        })
        .collect()
}

pub fn pairwise_fault_campaign() -> Vec<FaultCampaignCase> {
    let mut cases = Vec::new();
    for (i, a) in ALL_EVA_FAULTS.iter().copied().enumerate() {
        for b in ALL_EVA_FAULTS.iter().copied().skip(i + 1) {
            let scenario = FaultScenario::pair(a, b);
            let expected = scenario.disposition();
            cases.push(FaultCampaignCase { scenario, expected });
        }
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_fault_has_a_deterministic_disposition() {
        let campaign = single_fault_campaign();
        assert_eq!(campaign.len(), ALL_EVA_FAULTS.len());
        for case in campaign {
            assert_eq!(case.expected, case.scenario.disposition());
        }
    }

    #[test]
    fn all_pairwise_combinations_are_generated() {
        let n = ALL_EVA_FAULTS.len();
        assert_eq!(pairwise_fault_campaign().len(), n * (n - 1) / 2);
    }

    #[test]
    fn losing_both_oxygen_paths_aborts_eva() {
        let scenario = FaultScenario::pair(
            EvaFault::PrimaryOxygenRegulatorFailure,
            EvaFault::SecondaryOxygenRegulatorFailure,
        );
        assert_eq!(scenario.disposition(), EvaDisposition::AbortEva);
    }

    #[test]
    fn motor_short_removes_assist_not_life_support() {
        let scenario = FaultScenario::single(EvaFault::AssistMotorShort);
        assert_eq!(scenario.disposition(), EvaDisposition::DepowerAssist);
    }

    #[test]
    fn communication_loss_alone_preserves_local_autonomy() {
        let scenario = FaultScenario::single(EvaFault::CommunicationLoss);
        assert_eq!(scenario.disposition(), EvaDisposition::ContinueLocalAutonomy);
    }

    #[test]
    fn radiation_alert_dominates_other_faults() {
        for fault in ALL_EVA_FAULTS {
            let scenario = FaultScenario::pair(EvaFault::RadiationAlert, fault);
            assert_eq!(scenario.disposition(), EvaDisposition::ImmediateShelter);
        }
    }
}
