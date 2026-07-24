// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic certification evidence bundle.
//!
//! Digest providers are injected. The included deterministic implementation is
//! useful for reproducibility tests, but is not a cryptographic signature.

use crate::fault_tree::{BasicFault, FaultTreeEvaluation, TopEvent};
use crate::release_signoff::ReleaseGateReport;
use crate::requirements::RequirementRegistry;
use crate::safety_case::{SafetyCase, SafetyCaseAssessment};
use crate::scenario_manifest::ScenarioManifest;
use crate::scenario_runner::ScenarioRunReport;
use crate::traceability::{TraceabilityMatrix, TraceabilityReport};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CERTIFICATION_BUNDLE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildIdentity {
    pub source_tree: String,
    pub toolchain: String,
    pub dependency_profile: String,
    pub campaign_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CertificationBundle {
    pub schema_version: u16,
    pub system: String,
    pub build: BuildIdentity,
    pub registry: RequirementRegistry,
    pub manifests: Vec<ScenarioManifest>,
    pub scenario_reports: Vec<ScenarioRunReport>,
    pub traceability: TraceabilityMatrix,
    pub traceability_report: TraceabilityReport,
    pub fault_evaluation: FaultTreeEvaluation,
    pub minimal_cut_sets: BTreeMap<TopEvent, Vec<Vec<BasicFault>>>,
    pub safety_case: SafetyCase,
    pub safety_case_assessment: SafetyCaseAssessment,
    pub release_gate: ReleaseGateReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertificationBundleError {
    UnsupportedSchema(u16),
    EmptyBuildField(&'static str),
    InvalidRequirementRegistry,
    InvalidTraceability,
    IneligibleSafetyCase,
    IneligibleReleaseGate,
    DuplicateScenario(String),
    MissingScenarioReport(String),
    UnexpectedScenarioReport(String),
    ScenarioFingerprintMismatch(String),
    Serialization,
}

pub trait BundleDigestProvider {
    fn digest(&self, bytes: &[u8]) -> [u8; 32];
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DeterministicBundleDigest;

impl BundleDigestProvider for DeterministicBundleDigest {
    fn digest(&self, bytes: &[u8]) -> [u8; 32] {
        let mut lanes = [
            0x243f6a8885a308d3u64,
            0x13198a2e03707344,
            0xa4093822299f31d0,
            0x082efa98ec4e6c89,
        ];
        for (index, byte) in bytes.iter().enumerate() {
            let lane = index % lanes.len();
            lanes[lane] ^= u64::from(*byte).wrapping_add(index as u64);
            lanes[lane] = lanes[lane].rotate_left(((index + 11 * lane) % 63 + 1) as u32);
            lanes[lane] = lanes[lane].wrapping_mul(0x9e3779b185ebca87);
        }
        let mut digest = [0u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            digest[index * 8..(index + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        digest
    }
}

impl CertificationBundle {
    pub fn validate(&self) -> Result<(), CertificationBundleError> {
        if self.schema_version != CERTIFICATION_BUNDLE_SCHEMA_VERSION {
            return Err(CertificationBundleError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        for (label, value) in [
            ("source_tree", self.build.source_tree.as_str()),
            ("toolchain", self.build.toolchain.as_str()),
            ("dependency_profile", self.build.dependency_profile.as_str()),
            ("campaign_id", self.build.campaign_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CertificationBundleError::EmptyBuildField(label));
            }
        }
        if self.registry.validate().is_err() {
            return Err(CertificationBundleError::InvalidRequirementRegistry);
        }
        if !self.traceability_report.passes() {
            return Err(CertificationBundleError::InvalidTraceability);
        }
        if !self.safety_case_assessment.release_eligible() {
            return Err(CertificationBundleError::IneligibleSafetyCase);
        }
        if !self.release_gate.eligible() {
            return Err(CertificationBundleError::IneligibleReleaseGate);
        }
        let mut manifest_ids = BTreeSet::new();
        let mut manifest_fingerprints = BTreeMap::new();
        for manifest in &self.manifests {
            manifest.validate().map_err(|_| {
                CertificationBundleError::MissingScenarioReport(manifest.scenario_id.clone())
            })?;
            if !manifest_ids.insert(manifest.scenario_id.clone()) {
                return Err(CertificationBundleError::DuplicateScenario(
                    manifest.scenario_id.clone(),
                ));
            }
            let fingerprint = manifest.fingerprint().map_err(|_| {
                CertificationBundleError::MissingScenarioReport(manifest.scenario_id.clone())
            })?;
            manifest_fingerprints.insert(manifest.scenario_id.clone(), fingerprint);
        }
        let mut report_ids = BTreeSet::new();
        for report in &self.scenario_reports {
            if !report_ids.insert(report.scenario_id.clone()) {
                return Err(CertificationBundleError::DuplicateScenario(
                    report.scenario_id.clone(),
                ));
            }
            let Some(expected) = manifest_fingerprints.get(&report.scenario_id) else {
                return Err(CertificationBundleError::UnexpectedScenarioReport(
                    report.scenario_id.clone(),
                ));
            };
            if *expected != report.fingerprint {
                return Err(CertificationBundleError::ScenarioFingerprintMismatch(
                    report.scenario_id.clone(),
                ));
            }
        }
        for scenario_id in manifest_ids {
            if !report_ids.contains(&scenario_id) {
                return Err(CertificationBundleError::MissingScenarioReport(scenario_id));
            }
        }
        Ok(())
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn digest(
        &self,
        provider: &dyn BundleDigestProvider,
    ) -> Result<[u8; 32], CertificationBundleError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|_| CertificationBundleError::Serialization)?;
        Ok(provider.digest(json.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tree::FaultTreeModel;
    use crate::release_signoff::{ReleaseDecision, ReleaseGateReport};
    use crate::requirements::RequirementId;
    use crate::safety_case::SafetyCase;
    use crate::scenario_runner::ScenarioRunner;
    use std::collections::BTreeSet;

    fn bundle() -> CertificationBundle {
        let registry = RequirementRegistry::canonical();
        let manifest = ScenarioManifest::new(
            "nominal",
            "bundle scenario",
            2,
            vec![RequirementId::SafeCommandBounds],
        );
        let report = ScenarioRunner.run(&manifest).expect("valid scenario");
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
        CertificationBundle {
            schema_version: CERTIFICATION_BUNDLE_SCHEMA_VERSION,
            system: "symthaea-subterranean".into(),
            build: BuildIdentity {
                source_tree: "tree".into(),
                toolchain: "rust".into(),
                dependency_profile: "offline-compatible".into(),
                campaign_id: "campaign".into(),
            },
            registry,
            manifests: vec![manifest],
            scenario_reports: vec![report],
            traceability,
            traceability_report,
            fault_evaluation,
            minimal_cut_sets,
            safety_case,
            safety_case_assessment,
            release_gate: ReleaseGateReport {
                decision: ReleaseDecision::Eligible,
                blockers: Vec::new(),
                accepted_waivers: Vec::new(),
                distinct_approvers: 3,
            },
        }
    }

    #[test]
    fn bundle_validates_and_has_stable_digest() {
        let bundle = bundle();
        assert_eq!(bundle.validate(), Ok(()));
        let left = bundle.digest(&DeterministicBundleDigest).unwrap();
        let right = bundle.digest(&DeterministicBundleDigest).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn scenario_fingerprint_drift_is_rejected() {
        let mut bundle = bundle();
        bundle.scenario_reports[0].fingerprint.0[0] ^= 1;
        assert!(matches!(
            bundle.validate(),
            Err(CertificationBundleError::ScenarioFingerprintMismatch(_))
        ));
    }
}
