// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification check for ecological and civic stewardship obligations.
//!
//! `restoration_stewardship.rs`'s `RestorationLedger` already enforces the
//! properties that matter here -- completion requires both physical
//! progress and externally attributable evidence, and an overdue
//! obligation blocks new work and forces a return -- and is already
//! unit-tested for them. This module re-verifies the same properties as a
//! certification gate, exercising the real ledger rather than re-deriving
//! parallel stewardship logic, matching this crate's other certification
//! checks (see `long_horizon_validation.rs`, `stewardship_validation.rs`'s
//! siblings).

use crate::restoration_stewardship::{
    RestorationDisposition, RestorationError, RestorationLedger, RestorationObligation,
    RestorationObligationKind, RestorationState,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StewardshipGateFailure {
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StewardshipReport {
    pub failures: Vec<StewardshipGateFailure>,
}

impl StewardshipReport {
    pub fn passes(&self) -> bool {
        self.failures.is_empty()
    }
}

pub struct StewardshipValidator;

impl StewardshipValidator {
    fn obligation(id: u64, due_step: u64) -> RestorationObligation {
        RestorationObligation {
            id,
            kind: RestorationObligationKind::RestoreHabitat,
            node: None,
            required_quantity: 10.0,
            completed_quantity: 0.0,
            due_step,
            state: RestorationState::Open,
            authority_reference: "cert-stewardship-reference-permit".into(),
            completion_evidence_id: None,
            last_update_step: 0,
        }
    }

    fn check_evidence_required_for_completion(failures: &mut Vec<StewardshipGateFailure>) {
        let mut ledger = RestorationLedger::new();
        if ledger.add(Self::obligation(1, 100)).is_err() {
            failures.push(StewardshipGateFailure {
                detail: "reference obligation was rejected as invalid".into(),
            });
            return;
        }
        if let Err(error) = ledger.record_progress(1, 10.0, 3) {
            failures.push(StewardshipGateFailure {
                detail: format!("recording physical progress failed: {error:?}"),
            });
            return;
        }
        // Physical progress alone, attested without external verification,
        // must not be able to complete the obligation.
        let unverified = ledger.attest_completion(1, "self-reported", false, 4);
        if unverified != Err(RestorationError::InvalidEvidence) {
            failures.push(StewardshipGateFailure {
                detail: format!(
                    "unverified self-reported completion was not rejected, got {unverified:?}"
                ),
            });
        }
        if let Err(error) = ledger.attest_completion(1, "observer-evidence", true, 5) {
            failures.push(StewardshipGateFailure {
                detail: format!("externally verified completion was rejected: {error:?}"),
            });
            return;
        }
        if ledger.assess(5).disposition != RestorationDisposition::Clear {
            failures.push(StewardshipGateFailure {
                detail: "ledger did not clear after genuine, externally evidenced completion"
                    .into(),
            });
        }
    }

    fn check_overdue_blocks_work_and_requires_return(failures: &mut Vec<StewardshipGateFailure>) {
        let mut ledger = RestorationLedger::new();
        if ledger.add(Self::obligation(1, 10)).is_err() {
            failures.push(StewardshipGateFailure {
                detail: "reference obligation was rejected as invalid".into(),
            });
            return;
        }
        let assessment = ledger.assess(11);
        if assessment.disposition != RestorationDisposition::RestorationOverdue {
            failures.push(StewardshipGateFailure {
                detail: format!(
                    "overdue obligation produced {:?}, expected RestorationOverdue",
                    assessment.disposition
                ),
            });
        }
        if assessment.new_productive_work_allowed {
            failures.push(StewardshipGateFailure {
                detail: "overdue restoration obligation still allowed new productive work".into(),
            });
        }
        if !assessment.return_required {
            failures.push(StewardshipGateFailure {
                detail: "overdue restoration obligation did not require return".into(),
            });
        }
    }

    pub fn run(&self) -> StewardshipReport {
        let mut failures = Vec::new();
        Self::check_evidence_required_for_completion(&mut failures);
        Self::check_overdue_blocks_work_and_requires_return(&mut failures);
        StewardshipReport { failures }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restoration_obligations_are_genuinely_enforced() {
        let report = StewardshipValidator.run();
        assert!(report.passes(), "{report:#?}");
    }
}
