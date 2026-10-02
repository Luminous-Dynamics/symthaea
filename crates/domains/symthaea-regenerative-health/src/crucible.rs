// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic damage -> intervention -> verification crucible.
//!
//! The crucible is deliberately platform-neutral. Vehicle, marine, aerospace,
//! and industrial adapters can supply real observations later; this layer
//! verifies that the shared evidence contract never promotes ambiguous or
//! incomplete recovery into a trusted state.

use crate::{
    HealthObservation, RecoveryEvidence, RegenerativeAction, RegenerativeHealthGate,
    RegenerativeHealthState, RegenerativeIssue,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrucibleCase {
    Healthy,
    RepairPending,
    MissingVerification,
    ConfigurationMismatch,
    StaleRecovery,
    FutureObservation,
    UnqualifiedAction,
    Recovered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrucibleResult {
    pub case: CrucibleCase,
    pub state: RegenerativeHealthState,
    pub issues: Vec<RegenerativeIssue>,
}

pub fn run_standard_crucible(gate: &RegenerativeHealthGate) -> Vec<CrucibleResult> {
    let cases = [
        CrucibleCase::Healthy,
        CrucibleCase::RepairPending,
        CrucibleCase::MissingVerification,
        CrucibleCase::ConfigurationMismatch,
        CrucibleCase::StaleRecovery,
        CrucibleCase::FutureObservation,
        CrucibleCase::UnqualifiedAction,
        CrucibleCase::Recovered,
    ];

    cases.into_iter().map(|case| run_case(gate, case)).collect()
}

pub fn run_case(gate: &RegenerativeHealthGate, case: CrucibleCase) -> CrucibleResult {
    let observation = observation(case);
    let recovery = recovery(case);
    let action = action(case);

    let decision = gate.assess(&observation, recovery.as_ref(), action, 1_000);

    CrucibleResult {
        case,
        state: decision.state,
        issues: decision.issues,
    }
}

fn observation(case: CrucibleCase) -> HealthObservation {
    HealthObservation {
        observation_id: "crucible-observation".into(),
        component_id: "component-1".into(),
        timestamp_ms: if case == CrucibleCase::FutureObservation {
            1_001
        } else {
            1_000
        },
        normalized_residual: if matches!(
            case,
            CrucibleCase::RepairPending
                | CrucibleCase::MissingVerification
                | CrucibleCase::ConfigurationMismatch
                | CrucibleCase::StaleRecovery
                | CrucibleCase::Recovered
        ) {
            0.1
        } else {
            0.1
        },
        uncertainty: 1.0,
        evidence_ids: vec!["crucible-evidence".into()],
        configuration_digest: "cfg-1".into(),
    }
}

fn recovery(case: CrucibleCase) -> Option<RecoveryEvidence> {
    if !matches!(
        case,
        CrucibleCase::MissingVerification
            | CrucibleCase::ConfigurationMismatch
            | CrucibleCase::StaleRecovery
            | CrucibleCase::Recovered
    ) {
        return None;
    }

    Some(RecoveryEvidence {
        evidence_id: "recovery-1".into(),
        component_id: "component-1".into(),
        timestamp_ms: if case == CrucibleCase::StaleRecovery {
            0
        } else {
            1_000
        },
        normalized_residual: 0.1,
        uncertainty: 1.0,
        configuration_digest: if case == CrucibleCase::ConfigurationMismatch {
            "cfg-attacker".into()
        } else {
            "cfg-1".into()
        },
        intervention_id: "intervention-1".into(),
        independent_verification: case == CrucibleCase::Recovered,
    })
}

fn action(case: CrucibleCase) -> Option<RegenerativeAction> {
    Some(match case {
        CrucibleCase::Healthy => return None,
        CrucibleCase::UnqualifiedAction => RegenerativeAction::ReplaceModule,
        _ => RegenerativeAction::QualifiedRepair,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn gate() -> RegenerativeHealthGate {
        RegenerativeHealthGate::new(crate::RegenerativePolicy {
            schema_version: "0.1".into(),
            policy_id: "crucible-v1".into(),
            warning_sigma_milli: 2_000,
            restricted_sigma_milli: 4_000,
            recovery_sigma_milli: 1_500,
            maximum_observation_age_ms: 1_000,
            maximum_recovery_age_ms: 1_000,
            allowed_actions: [RegenerativeAction::QualifiedRepair]
                .into_iter()
                .collect::<BTreeSet<_>>(),
        })
        .expect("valid policy")
    }

    #[test]
    fn standard_crucible_preserves_recovery_boundary() {
        let results = run_standard_crucible(&gate());

        let recovered = results
            .iter()
            .find(|result| result.case == CrucibleCase::Recovered)
            .expect("recovered case");
        assert_eq!(recovered.state, RegenerativeHealthState::Recovered);

        for result in results.iter().filter(|result| {
            matches!(
                result.case,
                CrucibleCase::MissingVerification
                    | CrucibleCase::ConfigurationMismatch
                    | CrucibleCase::StaleRecovery
                    | CrucibleCase::FutureObservation
                    | CrucibleCase::UnqualifiedAction
            )
        }) {
            assert_ne!(result.state, RegenerativeHealthState::Recovered);
        }
    }

    #[test]
    fn future_evidence_is_quarantined() {
        let result = run_case(&gate(), CrucibleCase::FutureObservation);
        assert_eq!(result.state, RegenerativeHealthState::Quarantined);
        assert!(result
            .issues
            .contains(&RegenerativeIssue::FutureObservation));
    }

    #[test]
    fn configuration_mismatch_blocks_recovery() {
        let result = run_case(&gate(), CrucibleCase::ConfigurationMismatch);
        assert_ne!(result.state, RegenerativeHealthState::Recovered);
        assert!(result
            .issues
            .contains(&RegenerativeIssue::ConfigurationMismatch));
    }
}
