// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic certifiable-autonomy acceptance contracts.

use crate::actuator_isolation::ActuatorIsolationReport;
use crate::adaptation_validation::AdaptationValidator;
use crate::capability_profile::CapabilityDisposition;
use crate::certification_bundle::{
    BuildIdentity, CERTIFICATION_BUNDLE_SCHEMA_VERSION, CertificationBundle,
};
use crate::embodiment::MotorSafetyLevel;
use crate::fault_tree::{FaultTreeModel, TopEvent};
use crate::invariant_monitor::{InvariantContext, RuntimeInvariantMonitor};
use crate::lifecycle_validation::LifecycleAssuranceValidator;
use crate::release_signoff::{
    ReleaseGateInput, ReleaseSignoffGate, SignerId, SignerRole, VerifiedApproval,
};
use crate::requirements::{RequirementId, RequirementRegistry};
use crate::safety::SubterraneanHazard;
use crate::safety_case::SafetyCase;
use crate::scenario_manifest::{ScenarioManifest, StateOverride};
use crate::scenario_runner::ScenarioRunner;
use crate::stewardship_validation::StewardshipValidator;
use crate::traceability::TraceabilityMatrix;
use crate::types::{CUTTER_TEMP_C, SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificationContract {
    CanonicalRequirements,
    FinalCommandInvariant,
    BoundedFaultCuts,
    ReproducibleScenarios,
    CompleteTraceability,
    SupportedSafetyCase,
    RoleSeparatedRelease,
    SelfConsistentBundle,
    PostDeploymentLearning,
    LifecycleAssurance,
    EcologicalAndCivicStewardship,
}

impl CertificationContract {
    pub const ALL: [Self; 11] = [
        Self::CanonicalRequirements,
        Self::FinalCommandInvariant,
        Self::BoundedFaultCuts,
        Self::ReproducibleScenarios,
        Self::CompleteTraceability,
        Self::SupportedSafetyCase,
        Self::RoleSeparatedRelease,
        Self::SelfConsistentBundle,
        Self::PostDeploymentLearning,
        Self::LifecycleAssurance,
        Self::EcologicalAndCivicStewardship,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationGateFailure {
    pub contract: CertificationContract,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationValidationReport {
    pub passed: Vec<CertificationContract>,
    pub failures: Vec<CertificationGateFailure>,
}

impl CertificationValidationReport {
    pub fn passes(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CertificationValidator;

impl CertificationValidator {
    fn approvals() -> Vec<VerifiedApproval> {
        vec![
            VerifiedApproval {
                signer: SignerId(101),
                role: SignerRole::SafetyEngineer,
                hardware_backed: true,
            },
            VerifiedApproval {
                signer: SignerId(102),
                role: SignerRole::VerificationAuthority,
                hardware_backed: true,
            },
            VerifiedApproval {
                signer: SignerId(103),
                role: SignerRole::ReleaseManager,
                hardware_backed: true,
            },
        ]
    }

    fn thermal_manifest() -> ScenarioManifest {
        let mut manifest = ScenarioManifest::new(
            "cert-thermal-arrest",
            "certification thermal arrest",
            20,
            vec![
                RequirementId::HazardPreemption,
                RequirementId::SafeCommandBounds,
            ],
        );
        manifest.state_overrides.push(StateOverride {
            channel: CUTTER_TEMP_C,
            value: 150.0,
        });
        manifest
    }

    fn evaluate(contract: CertificationContract) -> Result<(), String> {
        match contract {
            CertificationContract::CanonicalRequirements => RequirementRegistry::canonical()
                .validate()
                .map_err(|error| format!("registry invalid: {error:?}")),
            CertificationContract::FinalCommandInvariant => {
                let state = SubterraneanState::home();
                let mut command = SubterraneanCommand::zero();
                command.set_cutter_head(1.0);
                command.set_auger_feed(1.0);
                command.set_thermal_pump(1.0);
                let (command, assessment) = RuntimeInvariantMonitor::default().enforce(
                    command,
                    InvariantContext {
                        state: &state,
                        safety_level: MotorSafetyLevel::Red,
                        primary_hazard: SubterraneanHazard::Thermal,
                        tunnel_conflict: false,
                        return_feasible: true,
                        capability_disposition: CapabilityDisposition::FullMission,
                        actuator_isolation: ActuatorIsolationReport::nominal(),
                    },
                );
                if command.cutter_head() != 0.0
                    || command.auger_feed() != 0.0
                    || command.thermal_pump() <= 0.0
                    || assessment.passed()
                {
                    return Err(
                        "final-command invariant did not remove work and preserve cooling".into(),
                    );
                }
                Ok(())
            }
            CertificationContract::BoundedFaultCuts => {
                let model = FaultTreeModel::canonical();
                if !model.validate() {
                    return Err("canonical fault tree failed structural validation".into());
                }
                for event in TopEvent::ALL {
                    if model.minimal_cut_sets(event).is_empty() {
                        return Err(format!("missing cut set for {event:?}"));
                    }
                }
                Ok(())
            }
            CertificationContract::ReproducibleScenarios => {
                let manifest = Self::thermal_manifest();
                let left = ScenarioRunner
                    .run(&manifest)
                    .map_err(|error| format!("{error:?}"))?;
                let right = ScenarioRunner
                    .run(&manifest)
                    .map_err(|error| format!("{error:?}"))?;
                if left != right || !left.passed() {
                    return Err(format!("scenario not reproducible or failed: {left:?}"));
                }
                Ok(())
            }
            CertificationContract::CompleteTraceability => {
                let registry = RequirementRegistry::canonical();
                let report = TraceabilityMatrix::canonical().validate(&registry, &[]);
                report
                    .passes()
                    .then_some(())
                    .ok_or_else(|| format!("{report:?}"))
            }
            CertificationContract::SupportedSafetyCase => {
                let registry = RequirementRegistry::canonical();
                let traceability = TraceabilityMatrix::canonical();
                let faults = FaultTreeModel::canonical().evaluate(&BTreeSet::new());
                let assessment =
                    SafetyCase::assemble(&registry, &traceability, &faults).assess(&registry);
                assessment
                    .release_eligible()
                    .then_some(())
                    .ok_or_else(|| format!("{assessment:?}"))
            }
            CertificationContract::RoleSeparatedRelease => {
                let registry = RequirementRegistry::canonical();
                let traceability_matrix = TraceabilityMatrix::canonical();
                let traceability = traceability_matrix.validate(&registry, &[]);
                let faults = FaultTreeModel::canonical().evaluate(&BTreeSet::new());
                let safety_case = SafetyCase::assemble(&registry, &traceability_matrix, &faults)
                    .assess(&registry);
                let scenario = ScenarioRunner
                    .run(&Self::thermal_manifest())
                    .map_err(|error| format!("{error:?}"))?;
                let approvals = Self::approvals();
                let report = ReleaseSignoffGate.evaluate(ReleaseGateInput {
                    registry: &registry,
                    traceability: &traceability,
                    safety_case: &safety_case,
                    scenarios: &[scenario],
                    waivers: &[],
                    release_approvals: &approvals,
                    evaluation_time_unix_seconds: 1,
                });
                report
                    .eligible()
                    .then_some(())
                    .ok_or_else(|| format!("{report:?}"))
            }
            CertificationContract::PostDeploymentLearning => {
                let report = AdaptationValidator.run();
                report
                    .passes()
                    .then_some(())
                    .ok_or_else(|| format!("{report:?}"))
            }
            CertificationContract::LifecycleAssurance => {
                let report = LifecycleAssuranceValidator.run();
                report
                    .passed()
                    .then_some(())
                    .ok_or_else(|| format!("{report:?}"))
            }
            CertificationContract::EcologicalAndCivicStewardship => {
                let report = StewardshipValidator.run();
                report
                    .passes()
                    .then_some(())
                    .ok_or_else(|| format!("{report:?}"))
            }
            CertificationContract::SelfConsistentBundle => {
                let registry = RequirementRegistry::canonical();
                let manifest = Self::thermal_manifest();
                let scenario = ScenarioRunner
                    .run(&manifest)
                    .map_err(|error| format!("{error:?}"))?;
                let traceability = TraceabilityMatrix::canonical();
                let traceability_report = traceability.validate(&registry, &[manifest.clone()]);
                let fault_model = FaultTreeModel::canonical();
                let fault_evaluation = fault_model.evaluate(&BTreeSet::new());
                let minimal_cut_sets = TopEvent::ALL
                    .into_iter()
                    .map(|event| (event, fault_model.minimal_cut_sets(event)))
                    .collect();
                let safety_case = SafetyCase::assemble(&registry, &traceability, &fault_evaluation);
                let safety_case_assessment = safety_case.assess(&registry);
                let approvals = Self::approvals();
                let release_gate = ReleaseSignoffGate.evaluate(ReleaseGateInput {
                    registry: &registry,
                    traceability: &traceability_report,
                    safety_case: &safety_case_assessment,
                    scenarios: std::slice::from_ref(&scenario),
                    waivers: &[],
                    release_approvals: &approvals,
                    evaluation_time_unix_seconds: 1,
                });
                let bundle = CertificationBundle {
                    schema_version: CERTIFICATION_BUNDLE_SCHEMA_VERSION,
                    system: "symthaea-subterranean".into(),
                    build: BuildIdentity {
                        source_tree: "acceptance-tree".into(),
                        toolchain: "offline-api-compatible".into(),
                        dependency_profile: "stand-in".into(),
                        campaign_id: "certification-validation".into(),
                    },
                    registry,
                    manifests: vec![manifest],
                    scenario_reports: vec![scenario],
                    traceability,
                    traceability_report,
                    fault_evaluation,
                    minimal_cut_sets,
                    safety_case,
                    safety_case_assessment,
                    release_gate,
                };
                bundle.validate().map_err(|error| format!("{error:?}"))
            }
        }
    }

    pub fn run(&self) -> CertificationValidationReport {
        let mut passed = Vec::new();
        let mut failures = Vec::new();
        for contract in CertificationContract::ALL {
            match Self::evaluate(contract) {
                Ok(()) => passed.push(contract),
                Err(detail) => failures.push(CertificationGateFailure { contract, detail }),
            }
        }
        CertificationValidationReport { passed, failures }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certifiable_autonomy_acceptance_contracts_pass() {
        let report = CertificationValidator.run();
        assert!(report.passes(), "{report:#?}");
        assert_eq!(report.passed, CertificationContract::ALL);
    }
}
