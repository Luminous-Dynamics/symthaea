// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit adapter from generic model assurance into domain-awareness ODD state.

#![deny(unsafe_code)]

use symthaea_domain_awareness::operational_domain::{
    ModelAssuranceState, OperationalConditions,
};
use symthaea_model_assurance::{
    ModelAssuranceError, ModelAssuranceReport, ModelAssuranceStatus,
};

/// Evidence-bound ODD model state derived from one model-assurance report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OddModelAssuranceEvidence {
    pub state: ModelAssuranceState,
    pub evidence_ref: String,
}

impl OddModelAssuranceEvidence {
    pub fn from_report(report: &ModelAssuranceReport) -> Result<Self, ModelAssuranceError> {
        let digest = report.digest_fnv1a64()?;
        Ok(Self {
            state: map_status(report.status),
            evidence_ref: format!(
                "model-assurance:{}:{}:{}",
                report.policy_id, report.assessed_at_ms, digest
            ),
        })
    }

    /// Apply the assurance state to already-constructed operating conditions.
    /// This adapter never upgrades any other ODD evidence and never grants authority.
    pub fn apply_to(&self, conditions: &mut OperationalConditions) {
        conditions.model_assurance = self.state;
        if !conditions.evidence_refs.contains(&self.evidence_ref) {
            conditions.evidence_refs.push(self.evidence_ref.clone());
        }
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub const fn map_status(status: ModelAssuranceStatus) -> ModelAssuranceState {
    match status {
        ModelAssuranceStatus::Aligned => ModelAssuranceState::Aligned,
        ModelAssuranceStatus::Restricted => ModelAssuranceState::Restricted,
        ModelAssuranceStatus::Unsafe => ModelAssuranceState::Unsafe,
        ModelAssuranceStatus::Incomplete => ModelAssuranceState::Incomplete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_domain_awareness::Modality;
    use symthaea_model_assurance::ModelAssuranceReport;

    fn report(status: ModelAssuranceStatus) -> ModelAssuranceReport {
        ModelAssuranceReport {
            schema_version: "1".into(),
            policy_id: "camera-model-v1".into(),
            assessed_at_ms: 1_000,
            status,
            signals: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn conditions() -> OperationalConditions {
        OperationalConditions {
            observed_at_ms: 1_000,
            visibility_m: Some(5_000.0),
            wind_speed_mps: Some(10.0),
            sea_state: None,
            navigation_quality: Some(0.9),
            available_modalities: vec![Modality::ElectroOptical],
            communications_available: Some(true),
            operator_available: Some(true),
            model_assurance: ModelAssuranceState::Aligned,
            evidence_refs: vec!["conditions:1".into()],
        }
    }

    #[test]
    fn every_generic_status_maps_explicitly() {
        assert_eq!(map_status(ModelAssuranceStatus::Aligned), ModelAssuranceState::Aligned);
        assert_eq!(map_status(ModelAssuranceStatus::Restricted), ModelAssuranceState::Restricted);
        assert_eq!(map_status(ModelAssuranceStatus::Unsafe), ModelAssuranceState::Unsafe);
        assert_eq!(map_status(ModelAssuranceStatus::Incomplete), ModelAssuranceState::Incomplete);
    }

    #[test]
    fn unsafe_report_cannot_become_aligned_in_adapter() {
        let evidence = OddModelAssuranceEvidence::from_report(&report(ModelAssuranceStatus::Unsafe))
            .unwrap();
        let mut current = conditions();
        evidence.apply_to(&mut current);
        assert_eq!(current.model_assurance, ModelAssuranceState::Unsafe);
        assert!(current.evidence_refs.iter().any(|value| value.starts_with("model-assurance:")));
        assert!(!evidence.grants_physical_authority());
    }

    #[test]
    fn evidence_application_is_idempotent() {
        let evidence = OddModelAssuranceEvidence::from_report(&report(ModelAssuranceStatus::Restricted))
            .unwrap();
        let mut current = conditions();
        evidence.apply_to(&mut current);
        evidence.apply_to(&mut current);
        assert_eq!(
            current
                .evidence_refs
                .iter()
                .filter(|value| *value == &evidence.evidence_ref)
                .count(),
            1
        );
    }
}
