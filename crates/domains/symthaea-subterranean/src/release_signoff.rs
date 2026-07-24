// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic release-signoff gate for certifiable autonomy artifacts.
//!
//! Identity authentication and signature verification are external. This
//! module consumes verified signer assertions and independently enforces role,
//! distinctness, waiver scope, expiry, and release-blocking semantics.

use crate::requirements::{RequirementCriticality, RequirementId, RequirementRegistry};
use crate::safety_case::SafetyCaseAssessment;
use crate::scenario_runner::ScenarioRunReport;
use crate::traceability::TraceabilityReport;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SignerId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignerRole {
    SafetyEngineer,
    VerificationAuthority,
    ReleaseManager,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedApproval {
    pub signer: SignerId,
    pub role: SignerRole,
    pub hardware_backed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementWaiver {
    pub requirement: RequirementId,
    pub rationale: String,
    pub expires_unix_seconds: u64,
    pub approvals: Vec<VerifiedApproval>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseBlocker {
    InvalidRequirementRegistry,
    TraceabilityIncomplete,
    SafetyCaseIneligible,
    ScenarioFailed(String),
    MissingScenarioEvidence,
    InvalidWaiver(RequirementId),
    UnwaivableRequirement(RequirementId),
    MissingReleaseApprovals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReleaseDecision {
    Eligible,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseGateReport {
    pub decision: ReleaseDecision,
    pub blockers: Vec<ReleaseBlocker>,
    pub accepted_waivers: Vec<RequirementId>,
    pub distinct_approvers: usize,
}

impl ReleaseGateReport {
    pub fn eligible(&self) -> bool {
        self.decision == ReleaseDecision::Eligible
    }
}

pub struct ReleaseGateInput<'a> {
    pub registry: &'a RequirementRegistry,
    pub traceability: &'a TraceabilityReport,
    pub safety_case: &'a SafetyCaseAssessment,
    pub scenarios: &'a [ScenarioRunReport],
    pub waivers: &'a [RequirementWaiver],
    pub release_approvals: &'a [VerifiedApproval],
    pub evaluation_time_unix_seconds: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReleaseSignoffGate;

impl ReleaseSignoffGate {
    fn waiver_valid(
        registry: &RequirementRegistry,
        waiver: &RequirementWaiver,
        now: u64,
    ) -> Result<(), ReleaseBlocker> {
        let Some(definition) = registry.definition(waiver.requirement) else {
            return Err(ReleaseBlocker::InvalidWaiver(waiver.requirement));
        };
        if definition.criticality == RequirementCriticality::Catastrophic {
            return Err(ReleaseBlocker::UnwaivableRequirement(waiver.requirement));
        }
        if waiver.rationale.trim().is_empty() || waiver.expires_unix_seconds <= now {
            return Err(ReleaseBlocker::InvalidWaiver(waiver.requirement));
        }
        let distinct_signers: BTreeSet<SignerId> = waiver
            .approvals
            .iter()
            .filter(|approval| approval.hardware_backed)
            .map(|approval| approval.signer)
            .collect();
        let roles: BTreeSet<SignerRole> = waiver
            .approvals
            .iter()
            .filter(|approval| approval.hardware_backed)
            .map(|approval| approval.role)
            .collect();
        if distinct_signers.len() < 2
            || !roles.contains(&SignerRole::SafetyEngineer)
            || !roles.contains(&SignerRole::VerificationAuthority)
        {
            return Err(ReleaseBlocker::InvalidWaiver(waiver.requirement));
        }
        Ok(())
    }

    fn release_approvals_valid(approvals: &[VerifiedApproval]) -> bool {
        let verified: Vec<&VerifiedApproval> = approvals
            .iter()
            .filter(|approval| approval.hardware_backed)
            .collect();
        let signers: BTreeSet<SignerId> = verified.iter().map(|approval| approval.signer).collect();
        let roles: BTreeSet<SignerRole> = verified.iter().map(|approval| approval.role).collect();
        signers.len() >= 3
            && roles.contains(&SignerRole::SafetyEngineer)
            && roles.contains(&SignerRole::VerificationAuthority)
            && roles.contains(&SignerRole::ReleaseManager)
    }

    pub fn evaluate(&self, input: ReleaseGateInput<'_>) -> ReleaseGateReport {
        let mut blockers = Vec::new();
        let mut accepted_waivers = Vec::new();
        if input.registry.validate().is_err() {
            blockers.push(ReleaseBlocker::InvalidRequirementRegistry);
        }
        if !input.traceability.passes() {
            blockers.push(ReleaseBlocker::TraceabilityIncomplete);
        }
        if !input.safety_case.release_eligible() {
            blockers.push(ReleaseBlocker::SafetyCaseIneligible);
        }
        if input.scenarios.is_empty() {
            blockers.push(ReleaseBlocker::MissingScenarioEvidence);
        }
        for scenario in input.scenarios {
            if !scenario.passed() {
                blockers.push(ReleaseBlocker::ScenarioFailed(scenario.scenario_id.clone()));
            }
        }
        for waiver in input.waivers {
            match Self::waiver_valid(input.registry, waiver, input.evaluation_time_unix_seconds) {
                Ok(()) => accepted_waivers.push(waiver.requirement),
                Err(blocker) => blockers.push(blocker),
            }
        }
        if !Self::release_approvals_valid(input.release_approvals) {
            blockers.push(ReleaseBlocker::MissingReleaseApprovals);
        }
        blockers.sort_by_key(|blocker| format!("{blocker:?}"));
        blockers.dedup();
        accepted_waivers.sort();
        accepted_waivers.dedup();
        let distinct_approvers = input
            .release_approvals
            .iter()
            .filter(|approval| approval.hardware_backed)
            .map(|approval| approval.signer)
            .collect::<BTreeSet<_>>()
            .len();
        ReleaseGateReport {
            decision: if blockers.is_empty() {
                ReleaseDecision::Eligible
            } else {
                ReleaseDecision::Blocked
            },
            blockers,
            accepted_waivers,
            distinct_approvers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tree::FaultTreeModel;
    use crate::safety_case::SafetyCase;
    use crate::traceability::TraceabilityMatrix;
    use std::collections::BTreeSet;

    fn approvals() -> Vec<VerifiedApproval> {
        vec![
            VerifiedApproval {
                signer: SignerId(1),
                role: SignerRole::SafetyEngineer,
                hardware_backed: true,
            },
            VerifiedApproval {
                signer: SignerId(2),
                role: SignerRole::VerificationAuthority,
                hardware_backed: true,
            },
            VerifiedApproval {
                signer: SignerId(3),
                role: SignerRole::ReleaseManager,
                hardware_backed: true,
            },
        ]
    }

    #[test]
    fn three_role_signoff_allows_clean_release() {
        let registry = RequirementRegistry::canonical();
        let traceability = TraceabilityMatrix::canonical().validate(&registry, &[]);
        let faults = FaultTreeModel::canonical().evaluate(&BTreeSet::new());
        let case = SafetyCase::assemble(&registry, &TraceabilityMatrix::canonical(), &faults)
            .assess(&registry);
        let scenario = ScenarioRunReport {
            scenario_id: "clean".into(),
            fingerprint: crate::scenario_manifest::ScenarioFingerprint([0; 32]),
            steps_executed: 1,
            final_state_valid: true,
            final_battery_ratio: 1.0,
            maximum_hazard_severity: 0.0,
            invariant_breach_records: 0,
            productive_work_at_red_records: 0,
            failures: Vec::new(),
        };
        let approvals = approvals();
        let report = ReleaseSignoffGate.evaluate(ReleaseGateInput {
            registry: &registry,
            traceability: &traceability,
            safety_case: &case,
            scenarios: &[scenario],
            waivers: &[],
            release_approvals: &approvals,
            evaluation_time_unix_seconds: 1,
        });
        assert!(report.eligible(), "{report:?}");
    }

    #[test]
    fn catastrophic_requirement_cannot_be_waived() {
        let registry = RequirementRegistry::canonical();
        let waiver = RequirementWaiver {
            requirement: RequirementId::HazardPreemption,
            rationale: "temporary".into(),
            expires_unix_seconds: 100,
            approvals: approvals(),
        };
        assert_eq!(
            ReleaseSignoffGate::waiver_valid(&registry, &waiver, 1),
            Err(ReleaseBlocker::UnwaivableRequirement(
                RequirementId::HazardPreemption
            ))
        );
    }
}
