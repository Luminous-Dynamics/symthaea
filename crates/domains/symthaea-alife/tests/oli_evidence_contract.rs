// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_alife::agnostic_evidence::{
    EvidenceBasis, EvidenceOutcome, EvidenceRecord, HypothesisClass, ObservationCoverage,
    Property, PropertyAssessment, PropertyFinding, SystemProfile, OLI_SCHEMA_VERSION,
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

fn measurement(id: &str, outcome: EvidenceOutcome) -> EvidenceRecord {
    EvidenceRecord {
        id: id.to_owned(),
        basis: EvidenceBasis::DirectMeasurement,
        outcome,
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

fn finding(property: Property, assessment: PropertyAssessment, evidence_ids: &[&str]) -> PropertyFinding {
    PropertyFinding {
        property,
        assessment,
        evidence_ids: evidence_ids.iter().map(|id| (*id).to_owned()).collect(),
        coverage: None,
        notes: "Scoped property assessment; not a global life verdict.".to_owned(),
    }
}

#[test]
fn property_vocabulary_round_trips_to_canonical_m0_values() {
    let values = [
        (PropertyAssessment::SupportedWithinScope, "SUPPORTED_WITHIN_SCOPE"),
        (PropertyAssessment::ContradictedWithinScope, "CONTRADICTED_WITHIN_SCOPE"),
        (PropertyAssessment::InsufficientEvidence, "INSUFFICIENT_EVIDENCE"),
        (PropertyAssessment::NotTested, "NOT_TESTED"),
        (
            PropertyAssessment::NotObservableWithCurrentInstrument,
            "NOT_OBSERVABLE_WITH_CURRENT_INSTRUMENT",
        ),
        (PropertyAssessment::ModelInadequate, "MODEL_INADEQUATE"),
    ];

    for (assessment, expected) in values {
        let encoded = serde_json::to_string(&assessment).expect("serializes");
        assert_eq!(encoded, format!("\"{expected}\""));
        let decoded: PropertyAssessment =
            serde_json::from_str(&encoded).expect("deserializes");
        assert_eq!(decoded, assessment);
    }
}

#[test]
fn non_detection_is_not_a_property_state_and_requires_coverage() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    candidate.findings.push(finding(
        Property::SelfMaintenance,
        PropertyAssessment::InsufficientEvidence,
        &["m1"],
    ));

    let errors = candidate.validate().expect_err("coverage is mandatory");
    assert!(errors.iter().any(|error| error.contains("requires coverage")));
}

#[test]
fn qualified_non_detection_remains_insufficient_evidence() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    let mut property = finding(
        Property::SelfMaintenance,
        PropertyAssessment::InsufficientEvidence,
        &["m1"],
    );
    property.coverage = Some(coverage());
    candidate.findings.push(property);

    assert!(candidate.validate().is_ok());
    assert_eq!(
        candidate.findings[0].assessment,
        PropertyAssessment::InsufficientEvidence
    );
}

#[test]
fn no_detection_cannot_be_promoted_to_positive_support() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    let mut property = finding(
        Property::Learning,
        PropertyAssessment::SupportedWithinScope,
        &["m1"],
    );
    property.coverage = Some(coverage());
    candidate.findings.push(property);

    let errors = candidate.validate().expect_err("negative-only evidence cannot support");
    assert!(errors
        .iter()
        .any(|error| error.contains("cannot be SupportedWithinScope solely from NoDetection")));
}

#[test]
fn observed_property_requires_a_linked_evidence_record() {
    let mut candidate = profile();
    candidate.findings.push(finding(
        Property::Adaptation,
        PropertyAssessment::SupportedWithinScope,
        &[],
    ));

    let errors = candidate.validate().expect_err("evidence reference is mandatory");
    assert!(errors.iter().any(|error| error.contains("requires at least one evidence record")));
}

#[test]
fn simulation_evidence_remains_distinct_from_measurement() {
    let mut candidate = profile();
    candidate.evidence.push(EvidenceRecord {
        id: "sim1".to_owned(),
        basis: EvidenceBasis::Simulation,
        outcome: EvidenceOutcome::SimulationResult,
        method: "molecular-dynamics simulation".to_owned(),
        source_ref: None,
        artifact_sha256: None,
        independence_group: None,
    });
    candidate.findings.push(finding(
        Property::InformationRetention,
        PropertyAssessment::SupportedWithinScope,
        &["sim1"],
    ));

    let encoded = serde_json::to_string(&candidate).expect("serializes");
    let decoded: SystemProfile = serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded.evidence[0].basis, EvidenceBasis::Simulation);
    assert!(decoded.validate().is_ok());
}

#[test]
fn not_observable_requires_an_instrument_and_coverage_record() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::InvalidMeasurement));
    candidate.findings.push(finding(
        Property::InformationTransfer,
        PropertyAssessment::NotObservableWithCurrentInstrument,
        &["m1"],
    ));

    let errors = candidate.validate().expect_err("coverage is mandatory");
    assert!(errors.iter().any(|error| error.contains("NotObservable requires observation coverage")));
}

#[test]
fn unfamiliar_substrate_labels_are_open_world() {
    let mut candidate = profile();
    candidate.substrate_family = "unknown field-coupled hybrid substrate".to_owned();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::SignalDetected));
    candidate.findings.push(finding(
        Property::Boundary,
        PropertyAssessment::SupportedWithinScope,
        &["m1"],
    ));

    assert!(candidate.validate().is_ok());
}

#[test]
fn alternative_hypotheses_must_make_discriminating_predictions() {
    let mut candidate = profile();
    candidate.alternative_hypotheses.push(
        symthaea_alife::agnostic_evidence::AlternativeHypothesis {
            id: "abiotic-null".to_owned(),
            class: HypothesisClass::AbioticPhysical,
            mechanism: "reaction-diffusion instability".to_owned(),
            discriminating_predictions: Vec::new(),
            evidence_ids: Vec::new(),
        },
    );

    let errors = candidate.validate().expect_err("predictions are required");
    assert!(errors
        .iter()
        .any(|error| error.contains("requires non-empty discriminating predictions")));
}
