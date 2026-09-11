// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic suitport/service state machine for SX-019.
//!
//! The state machine produces pressure-boundary permission only after local
//! identity, capture, seal, pressure, utility, decontamination, transfer, and
//! diagnostic evidence has been satisfied in order. No AI/planner request can
//! skip a transition or directly open the pressure boundary.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuitportStage {
    Approach,
    Capture,
    SealVerify,
    PressureBoundaryVerify,
    UtilityNegotiate,
    DustDecontamination,
    ResourceTransfer,
    Diagnostics,
    DonDoffReady,
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuitportFault {
    IdentityInvalid,
    CaptureLost,
    SealInvalid,
    PressureEvidenceInvalid,
    UtilityEvidenceInvalid,
    DecontaminationIncomplete,
    TransferIncomplete,
    DiagnosticsFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitportEvidence {
    pub identity_verified: bool,
    pub mechanical_capture_locked: bool,
    pub outer_seal_verified: bool,
    pub inner_seal_verified: bool,
    pub pressure_sensors_agree: bool,
    pub pressure_difference_within_limit: bool,
    pub required_utilities_negotiated: bool,
    pub dust_decontamination_complete: bool,
    pub resource_transfer_complete: bool,
    pub diagnostics_passed: bool,
    pub evidence: ExosuitEvidenceLevel,
}

impl SuitportEvidence {
    pub fn simulation_nominal() -> Self {
        Self {
            identity_verified: true,
            mechanical_capture_locked: true,
            outer_seal_verified: true,
            inner_seal_verified: true,
            pressure_sensors_agree: true,
            pressure_difference_within_limit: true,
            required_utilities_negotiated: true,
            dust_decontamination_complete: true,
            resource_transfer_complete: true,
            diagnostics_passed: true,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitportTransition {
    pub previous: SuitportStage,
    pub current: SuitportStage,
    pub fault: Option<SuitportFault>,
    /// True only in `DonDoffReady` after all ordered checks passed.
    pub pressure_boundary_open_permitted: bool,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuitportStateMachine {
    stage: SuitportStage,
    fault: Option<SuitportFault>,
    evidence: ExosuitEvidenceLevel,
}

impl SuitportStateMachine {
    pub fn simulation_reference() -> Self {
        Self {
            stage: SuitportStage::Approach,
            fault: None,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn stage(&self) -> SuitportStage {
        self.stage
    }

    pub fn fault(&self) -> Option<SuitportFault> {
        self.fault
    }

    pub fn pressure_boundary_open_permitted(&self) -> bool {
        self.stage == SuitportStage::DonDoffReady && self.fault.is_none()
    }

    /// Advance at most one ordered stage using local evidence.
    pub fn advance(&mut self, evidence: SuitportEvidence) -> SuitportTransition {
        let previous = self.stage;

        if self.stage == SuitportStage::Faulted {
            return self.transition(previous);
        }

        if !evidence.identity_verified {
            self.fail(SuitportFault::IdentityInvalid);
            return self.transition(previous);
        }

        // Once physical capture is expected, losing it is a hard docking fault.
        if self.stage >= SuitportStage::Capture && !evidence.mechanical_capture_locked {
            self.fail(SuitportFault::CaptureLost);
            return self.transition(previous);
        }

        self.stage = match self.stage {
            SuitportStage::Approach => {
                if evidence.mechanical_capture_locked {
                    SuitportStage::Capture
                } else {
                    SuitportStage::Approach
                }
            }
            SuitportStage::Capture => {
                if evidence.outer_seal_verified && evidence.inner_seal_verified {
                    SuitportStage::SealVerify
                } else {
                    self.fail(SuitportFault::SealInvalid);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::SealVerify => {
                if evidence.pressure_sensors_agree && evidence.pressure_difference_within_limit {
                    SuitportStage::PressureBoundaryVerify
                } else {
                    self.fail(SuitportFault::PressureEvidenceInvalid);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::PressureBoundaryVerify => {
                if evidence.required_utilities_negotiated {
                    SuitportStage::UtilityNegotiate
                } else {
                    self.fail(SuitportFault::UtilityEvidenceInvalid);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::UtilityNegotiate => {
                if evidence.dust_decontamination_complete {
                    SuitportStage::DustDecontamination
                } else {
                    self.fail(SuitportFault::DecontaminationIncomplete);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::DustDecontamination => {
                if evidence.resource_transfer_complete {
                    SuitportStage::ResourceTransfer
                } else {
                    self.fail(SuitportFault::TransferIncomplete);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::ResourceTransfer => {
                if evidence.diagnostics_passed {
                    SuitportStage::Diagnostics
                } else {
                    self.fail(SuitportFault::DiagnosticsFailed);
                    SuitportStage::Faulted
                }
            }
            SuitportStage::Diagnostics => SuitportStage::DonDoffReady,
            SuitportStage::DonDoffReady => SuitportStage::DonDoffReady,
            SuitportStage::Faulted => SuitportStage::Faulted,
        };

        self.evidence = self.evidence.min(evidence.evidence);
        self.transition(previous)
    }

    /// Explicitly reset a faulted/finished docking sequence back to approach.
    /// This is a simulation state reset, not a pressure-boundary command.
    pub fn reset_to_approach(&mut self) {
        self.stage = SuitportStage::Approach;
        self.fault = None;
    }

    fn fail(&mut self, fault: SuitportFault) {
        self.stage = SuitportStage::Faulted;
        self.fault = Some(fault);
    }

    fn transition(&self, previous: SuitportStage) -> SuitportTransition {
        SuitportTransition {
            previous,
            current: self.stage,
            fault: self.fault,
            pressure_boundary_open_permitted: self.pressure_boundary_open_permitted(),
            evidence: self.evidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_sequence_reaches_boundary_permission_only_at_end() {
        let mut machine = SuitportStateMachine::simulation_reference();
        let evidence = SuitportEvidence::simulation_nominal();
        for _ in 0..7 {
            let transition = machine.advance(evidence);
            assert!(!transition.pressure_boundary_open_permitted);
        }
        let transition = machine.advance(evidence);
        assert_eq!(machine.stage(), SuitportStage::DonDoffReady);
        assert!(transition.pressure_boundary_open_permitted);
    }

    #[test]
    fn decon_cannot_be_skipped_before_transfer() {
        let mut machine = SuitportStateMachine::simulation_reference();
        let nominal = SuitportEvidence::simulation_nominal();
        for _ in 0..4 {
            machine.advance(nominal);
        }
        assert_eq!(machine.stage(), SuitportStage::UtilityNegotiate);

        let mut dirty = nominal;
        dirty.dust_decontamination_complete = false;
        let transition = machine.advance(dirty);
        assert_eq!(transition.current, SuitportStage::Faulted);
        assert_eq!(transition.fault, Some(SuitportFault::DecontaminationIncomplete));
        assert!(!transition.pressure_boundary_open_permitted);
    }

    #[test]
    fn capture_loss_after_capture_fails_closed() {
        let mut machine = SuitportStateMachine::simulation_reference();
        let nominal = SuitportEvidence::simulation_nominal();
        machine.advance(nominal);
        let mut lost = nominal;
        lost.mechanical_capture_locked = false;
        let transition = machine.advance(lost);
        assert_eq!(transition.fault, Some(SuitportFault::CaptureLost));
        assert!(!transition.pressure_boundary_open_permitted);
    }
}
