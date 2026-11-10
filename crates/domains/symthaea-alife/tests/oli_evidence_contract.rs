// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_alife::agnostic_evidence::{
    AlternativeHypothesis, EvidenceBasis, EvidenceRecord, EvidenceState, HypothesisClass,
    ObservationCoverage, Property, PropertyFinding, SystemProfile, OLI_SCHEMA_VERSION,
};

fn profile() -> SystemProfile {
    SystemProfile {
        schema_version: OLI_SCHEMA_VERSION.to_owned(),
        system_id: "case:magnetized-discharge-001".to_owned(),
        substrate_family: "strongly magnetized low-pressure electric discharge".to_owned(),
        organizational_scale: "filamentary pattern".to_owned(),
        findings: Vec::new(),
        evidence: Vec::new(),
        alternative_hypotheses: Vec::new(),
    }
}

fn measurement(id: &str) -> EvidenceRecord {
    EvidenceRecord {
        id: id.to_owned(),
        basis: EvidenceBasis::DirectMeasurement,
        method: "calibrated optical imaging".to_owned(),
        source_ref: Some("https://doi.org/10.1038/s42005-023-01337-3".to_owned()),
        artifact_sha256: None,
        independence_group: Some("experiment-series-a".to_owned()),
    }
}

fn coverage() -> ObservationCoverage {
    ObservationCoverage {
        modality: "time-resolved optical imaging".to_owned(),
        scope: "field of view covering 80% of the discharge chamber".to_owned(),
        observation_duration_seconds: Some(30.0),
        spatial_coverage_fraction: Some(0.8),
        detection_limit: Some(0.01),
    }
}

#[test]
fn non_detection_requires_measurement_and_coverage() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1"));
    candidate.findings.push(PropertyFinding {
        property: Property::SelfProduction,
        state: EvidenceState::NotDetected,
        evidence_ids: vec!["m1".to_owned()],
        coverage: None,
        notes: "No self-production observed in this measurement window.".to_owned(),
    });

    let errors = candidate.validate().expect_err("coverage is mandatory");
    assert!(errors.iter().any(|error| error.contains("requires observation coverage")));
}

#[test]
fn qualified_non_detection_is_not_an_absence_claim() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1"));
    candidate.findings.push(PropertyFinding {
        property: Property::SelfProduction,
        state: EvidenceState::NotDetected,
        evidence_ids: vec!["m1".to_owned()],
        coverage: Some(coverage()),
        notes: "Not detected within the stated spatial and temporal coverage.".to_owned(),
    });

    assert!(candidate.validate().is_ok());
    assert_eq!(candidate.findings[0].state, EvidenceState::NotDetected);
}

#[test]
fn observed_property_requires_a_linked_evidence_record() {
    let mut candidate = profile();
    candidate.findings.push(PropertyFinding {
        property: Property::AdaptiveResponse,
        state: EvidenceState::Observed,
        evidence_ids: Vec::new(),
        coverage: None,
        notes: "Unreferenced assertion must fail.".to_owned(),
    });

    let errors = candidate.validate().expect_err("evidence reference is mandatory");
    assert!(errors.iter().any(|error| error.contains("requires at least one evidence record")));
}

#[test]
fn simulation_evidence_remains_distinct_from_measurement() {
    let mut candidate = profile();
    candidate.evidence.push(EvidenceRecord {
        id: "sim1".to_owned(),
        basis: EvidenceBasis::Simulation,
        method: "molecular-dynamics simulation".to_owned(),
        source_ref: None,
        artifact_sha256: None,
        independence_group: None,
    });
    candidate.findings.push(PropertyFinding {
        property: Property::InformationPersistence,
        state: EvidenceState::Observed,
        evidence_ids: vec!["sim1".to_owned()],
        coverage: None,
        notes: "Observed in simulation only; not a laboratory measurement.".to_owned(),
    });

    let encoded = serde_json::to_string(&candidate).expect("serializes");
    let decoded: SystemProfile = serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded.evidence[0].basis, EvidenceBasis::Simulation);
    assert!(decoded.validate().is_ok());
}

#[test]
fn conflicted_finding_requires_multiple_evidence_records() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1"));
    candidate.findings.push(PropertyFinding {
        property: Property::Learning,
        state: EvidenceState::Conflicted,
        evidence_ids: vec!["m1".to_owned()],
        coverage: None,
        notes: "A conflict cannot be represented by one record.".to_owned(),
    });

    let errors = candidate.validate().expect_err("two records are required");
    assert!(errors.iter().any(|error| error.contains("requires at least two evidence records")));
}

#[test]
fn unfamiliar_substrate_labels_are_open_world() {
    let mut candidate = profile();
    candidate.substrate_family = "unknown field-coupled hybrid substrate".to_owned();
    candidate.evidence.push(measurement("m1"));
    candidate.findings.push(PropertyFinding {
        property: Property::BoundaryMaintenance,
        state: EvidenceState::Observed,
        evidence_ids: vec!["m1".to_owned()],
        coverage: None,
        notes: "A property observation does not imply life.".to_owned(),
    });

    assert!(candidate.validate().is_ok());
}

#[test]
fn alternative_hypotheses_must_make_discriminating_predictions() {
    let mut candidate = profile();
    candidate.alternative_hypotheses.push(AlternativeHypothesis {
        id: "abiotic-null".to_owned(),
        class: HypothesisClass::AbioticPhysical,
        mechanism: "reaction-diffusion instability".to_owned(),
        discriminating_predictions: Vec::new(),
        evidence_ids: Vec::new(),
    });

    let errors = candidate.validate().expect_err("predictions are required");
    assert!(errors.iter().any(|error| error.contains("requires non-empty discriminating predictions")));
}
