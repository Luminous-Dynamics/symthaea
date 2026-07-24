// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fleet configuration-drift detection.
//!
//! Qualification belongs to a concrete aircraft configuration, not merely to a
//! source revision. This module compares each aircraft snapshot with a fleet
//! baseline and distinguishes permitted variance from restricted or grounding
//! drift. Missing identity and evidence are Incomplete rather than silently
//! accepted.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetConfigurationBaseline {
    pub schema_version: String,
    pub baseline_id: String,
    pub fleet_id: String,
    pub deployment_digest: String,
    pub calibration_digest: String,
    pub hardware_contract_digest: String,
    pub required_modules: BTreeMap<String, String>,
    pub required_parameter_digests: BTreeMap<String, String>,
    pub qualification_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AircraftConfigurationSnapshot {
    pub aircraft_id: String,
    pub fleet_id: String,
    pub captured_at_ms: u64,
    pub deployment_digest: String,
    pub calibration_digest: String,
    pub hardware_contract_digest: String,
    pub installed_modules: BTreeMap<String, String>,
    pub parameter_digests: BTreeMap<String, String>,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetDriftPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub maximum_snapshot_age_ms: u64,
    pub permitted_module_versions: BTreeMap<String, Vec<String>>,
    pub permitted_parameter_digests: BTreeMap<String, Vec<String>>,
    pub restricted_module_names: Vec<String>,
    pub grounding_module_names: Vec<String>,
    pub require_snapshot_evidence: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AircraftDriftStatus {
    Compliant,
    Restricted,
    Grounded,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetDriftIssue {
    InvalidIdentity,
    FleetMismatch,
    StaleSnapshot {
        age_ms: u64,
        maximum_ms: u64,
    },
    MissingEvidence,
    DeploymentMismatch,
    CalibrationMismatch,
    HardwareContractMismatch,
    MissingModule(String),
    UnexpectedModule(String),
    ModuleVersionMismatch {
        module: String,
        expected: String,
        observed: String,
    },
    MissingParameter(String),
    ParameterDigestMismatch {
        parameter: String,
        expected: String,
        observed: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AircraftDriftAssessment {
    pub aircraft_id: String,
    pub baseline_id: String,
    pub status: AircraftDriftStatus,
    pub issues: Vec<FleetDriftIssue>,
    pub snapshot_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetDriftReport {
    pub fleet_id: String,
    pub baseline_id: String,
    pub assessed_at_ms: u64,
    pub status: AircraftDriftStatus,
    pub aircraft: Vec<AircraftDriftAssessment>,
    pub compliant_count: usize,
    pub restricted_count: usize,
    pub grounded_count: usize,
    pub incomplete_count: usize,
}

impl FleetDriftReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, FleetDriftError> {
        let mut canonical = self.clone();
        canonical
            .aircraft
            .sort_by(|a, b| a.aircraft_id.cmp(&b.aircraft_id));
        serde_json::to_vec(&canonical).map_err(|_| FleetDriftError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, FleetDriftError> {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.canonical_json()? {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FleetDriftError {
    InvalidBaseline,
    InvalidPolicy,
    DuplicateAircraft(String),
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct FleetDriftMonitor {
    baseline: FleetConfigurationBaseline,
    policy: FleetDriftPolicy,
}

impl FleetDriftMonitor {
    pub fn new(
        baseline: FleetConfigurationBaseline,
        policy: FleetDriftPolicy,
    ) -> Result<Self, FleetDriftError> {
        if baseline.schema_version.trim().is_empty()
            || baseline.baseline_id.trim().is_empty()
            || baseline.fleet_id.trim().is_empty()
            || !valid_digest(&baseline.deployment_digest)
            || !valid_digest(&baseline.calibration_digest)
            || !valid_digest(&baseline.hardware_contract_digest)
            || baseline.required_modules.is_empty()
        {
            return Err(FleetDriftError::InvalidBaseline);
        }
        let restricted: BTreeSet<_> = policy.restricted_module_names.iter().collect();
        let grounding: BTreeSet<_> = policy.grounding_module_names.iter().collect();
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.maximum_snapshot_age_ms == 0
            || restricted.len() != policy.restricted_module_names.len()
            || grounding.len() != policy.grounding_module_names.len()
            || !restricted.is_disjoint(&grounding)
        {
            return Err(FleetDriftError::InvalidPolicy);
        }
        Ok(Self { baseline, policy })
    }

    pub fn assess_aircraft(
        &self,
        snapshot: &AircraftConfigurationSnapshot,
        now_ms: u64,
    ) -> AircraftDriftAssessment {
        let mut issues = Vec::new();
        if snapshot.aircraft_id.trim().is_empty() || snapshot.fleet_id.trim().is_empty() {
            issues.push(FleetDriftIssue::InvalidIdentity);
        }
        if snapshot.fleet_id != self.baseline.fleet_id {
            issues.push(FleetDriftIssue::FleetMismatch);
        }
        let age_ms = now_ms.saturating_sub(snapshot.captured_at_ms);
        if age_ms > self.policy.maximum_snapshot_age_ms {
            issues.push(FleetDriftIssue::StaleSnapshot {
                age_ms,
                maximum_ms: self.policy.maximum_snapshot_age_ms,
            });
        }
        if self.policy.require_snapshot_evidence && snapshot.evidence_ids.is_empty() {
            issues.push(FleetDriftIssue::MissingEvidence);
        }
        if snapshot.deployment_digest != self.baseline.deployment_digest {
            issues.push(FleetDriftIssue::DeploymentMismatch);
        }
        if snapshot.calibration_digest != self.baseline.calibration_digest {
            issues.push(FleetDriftIssue::CalibrationMismatch);
        }
        if snapshot.hardware_contract_digest != self.baseline.hardware_contract_digest {
            issues.push(FleetDriftIssue::HardwareContractMismatch);
        }

        for (module, expected) in &self.baseline.required_modules {
            let Some(observed) = snapshot.installed_modules.get(module) else {
                issues.push(FleetDriftIssue::MissingModule(module.clone()));
                continue;
            };
            if observed != expected
                && !self
                    .policy
                    .permitted_module_versions
                    .get(module)
                    .is_some_and(|versions| versions.contains(observed))
            {
                issues.push(FleetDriftIssue::ModuleVersionMismatch {
                    module: module.clone(),
                    expected: expected.clone(),
                    observed: observed.clone(),
                });
            }
        }
        for module in snapshot.installed_modules.keys() {
            if !self.baseline.required_modules.contains_key(module) {
                issues.push(FleetDriftIssue::UnexpectedModule(module.clone()));
            }
        }
        for (parameter, expected) in &self.baseline.required_parameter_digests {
            let Some(observed) = snapshot.parameter_digests.get(parameter) else {
                issues.push(FleetDriftIssue::MissingParameter(parameter.clone()));
                continue;
            };
            if observed != expected
                && !self
                    .policy
                    .permitted_parameter_digests
                    .get(parameter)
                    .is_some_and(|digests| digests.contains(observed))
            {
                issues.push(FleetDriftIssue::ParameterDigestMismatch {
                    parameter: parameter.clone(),
                    expected: expected.clone(),
                    observed: observed.clone(),
                });
            }
        }

        let status = classify(&issues, &self.policy);
        AircraftDriftAssessment {
            aircraft_id: snapshot.aircraft_id.clone(),
            baseline_id: self.baseline.baseline_id.clone(),
            status,
            issues,
            snapshot_age_ms: age_ms,
        }
    }

    pub fn assess_fleet(
        &self,
        snapshots: &[AircraftConfigurationSnapshot],
        now_ms: u64,
    ) -> Result<FleetDriftReport, FleetDriftError> {
        let mut ids = BTreeSet::new();
        for snapshot in snapshots {
            if !ids.insert(snapshot.aircraft_id.clone()) {
                return Err(FleetDriftError::DuplicateAircraft(
                    snapshot.aircraft_id.clone(),
                ));
            }
        }
        let aircraft: Vec<_> = snapshots
            .iter()
            .map(|snapshot| self.assess_aircraft(snapshot, now_ms))
            .collect();
        let compliant_count = aircraft
            .iter()
            .filter(|a| a.status == AircraftDriftStatus::Compliant)
            .count();
        let restricted_count = aircraft
            .iter()
            .filter(|a| a.status == AircraftDriftStatus::Restricted)
            .count();
        let grounded_count = aircraft
            .iter()
            .filter(|a| a.status == AircraftDriftStatus::Grounded)
            .count();
        let incomplete_count = aircraft
            .iter()
            .filter(|a| a.status == AircraftDriftStatus::Incomplete)
            .count();
        let status = if grounded_count > 0 {
            AircraftDriftStatus::Grounded
        } else if incomplete_count > 0 {
            AircraftDriftStatus::Incomplete
        } else if restricted_count > 0 {
            AircraftDriftStatus::Restricted
        } else {
            AircraftDriftStatus::Compliant
        };
        Ok(FleetDriftReport {
            fleet_id: self.baseline.fleet_id.clone(),
            baseline_id: self.baseline.baseline_id.clone(),
            assessed_at_ms: now_ms,
            status,
            aircraft,
            compliant_count,
            restricted_count,
            grounded_count,
            incomplete_count,
        })
    }
}

fn classify(issues: &[FleetDriftIssue], policy: &FleetDriftPolicy) -> AircraftDriftStatus {
    if issues.iter().any(|issue| match issue {
        FleetDriftIssue::DeploymentMismatch
        | FleetDriftIssue::CalibrationMismatch
        | FleetDriftIssue::HardwareContractMismatch
        | FleetDriftIssue::FleetMismatch
        | FleetDriftIssue::InvalidIdentity => true,
        FleetDriftIssue::MissingModule(module)
        | FleetDriftIssue::ModuleVersionMismatch { module, .. }
        | FleetDriftIssue::UnexpectedModule(module) => {
            policy.grounding_module_names.contains(module)
        }
        _ => false,
    }) {
        AircraftDriftStatus::Grounded
    } else if issues.iter().any(|issue| {
        matches!(
            issue,
            FleetDriftIssue::StaleSnapshot { .. }
                | FleetDriftIssue::MissingEvidence
                | FleetDriftIssue::MissingParameter(_)
        )
    }) {
        AircraftDriftStatus::Incomplete
    } else if issues.is_empty() {
        AircraftDriftStatus::Compliant
    } else {
        AircraftDriftStatus::Restricted
    }
}

fn valid_digest(digest: &str) -> bool {
    let Some((algorithm, value)) = digest.split_once(':') else {
        return false;
    };
    !algorithm.is_empty() && value.len() >= 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor() -> FleetDriftMonitor {
        FleetDriftMonitor::new(
            FleetConfigurationBaseline {
                schema_version: "1".into(),
                baseline_id: "baseline-a".into(),
                fleet_id: "sar".into(),
                deployment_digest: "sha256:1111111111111111".into(),
                calibration_digest: "sha256:2222222222222222".into(),
                hardware_contract_digest: "sha256:3333333333333333".into(),
                required_modules: BTreeMap::from([
                    ("flight-control".into(), "2.0".into()),
                    ("recorder".into(), "1.0".into()),
                ]),
                required_parameter_digests: BTreeMap::from([(
                    "rotor".into(),
                    "sha256:4444444444444444".into(),
                )]),
                qualification_evidence_ids: vec!["qual".into()],
            },
            FleetDriftPolicy {
                schema_version: "1".into(),
                policy_id: "fleet-drift".into(),
                maximum_snapshot_age_ms: 1_000,
                permitted_module_versions: BTreeMap::new(),
                permitted_parameter_digests: BTreeMap::new(),
                restricted_module_names: vec!["recorder".into()],
                grounding_module_names: vec!["flight-control".into()],
                require_snapshot_evidence: true,
            },
        )
        .unwrap()
    }

    fn snapshot() -> AircraftConfigurationSnapshot {
        AircraftConfigurationSnapshot {
            aircraft_id: "aircraft-1".into(),
            fleet_id: "sar".into(),
            captured_at_ms: 100,
            deployment_digest: "sha256:1111111111111111".into(),
            calibration_digest: "sha256:2222222222222222".into(),
            hardware_contract_digest: "sha256:3333333333333333".into(),
            installed_modules: BTreeMap::from([
                ("flight-control".into(), "2.0".into()),
                ("recorder".into(), "1.0".into()),
            ]),
            parameter_digests: BTreeMap::from([("rotor".into(), "sha256:4444444444444444".into())]),
            evidence_ids: vec!["snapshot".into()],
        }
    }

    #[test]
    fn exact_snapshot_is_compliant() {
        assert_eq!(
            monitor().assess_aircraft(&snapshot(), 200).status,
            AircraftDriftStatus::Compliant
        );
    }

    #[test]
    fn flight_control_drift_grounds_aircraft() {
        let mut snapshot = snapshot();
        snapshot
            .installed_modules
            .insert("flight-control".into(), "unknown".into());
        assert_eq!(
            monitor().assess_aircraft(&snapshot, 200).status,
            AircraftDriftStatus::Grounded
        );
    }

    #[test]
    fn stale_snapshot_is_incomplete() {
        assert_eq!(
            monitor().assess_aircraft(&snapshot(), 2_000).status,
            AircraftDriftStatus::Incomplete
        );
    }
}
