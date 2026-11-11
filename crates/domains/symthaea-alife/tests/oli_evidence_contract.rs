// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_alife::agnostic_evidence::{
    EvidenceBasis, EvidenceOutcome, EvidenceRecord, FalseNegativeRisk, HypothesisClass, ObservationCoverage,
    Property, PropertyAssessment, PropertyFinding, SystemProfile, SYMTHAEA_OLI_CONTEXT_VERSION,
};

fn profile() -> SystemProfile {
    SystemProfile {
        schema_version: SYMTHAEA_OLI_CONTEXT_VERSION.to_owned(),
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
        temporal_coverage_fraction: Some(0.6),
        temporal_sampling_interval_seconds: Some(2.0),
        target_process_timescale_seconds: Some(0.5),
        detection_limit: Some(0.01),
        detection_limit_units: Some("relative signal units".to_owned()),
        reference_abundance: None,
        detection_probability_at_reference: None,
        calibration_evidence_id: None,
        false_negative_risks: vec![
            FalseNegativeRisk::LowAbundance,
            FalseNegativeRisk::TemporalAliasing,
        ],
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

fn model_diagnostic(id: &str) -> EvidenceRecord {
    EvidenceRecord {
        id: id.to_owned(),
        basis: EvidenceBasis::DerivedInference,
        outcome: EvidenceOutcome::DerivedResult,
        method: "posterior-predictive residual adequacy check".to_owned(),
        source_ref: Some("run://model-adequacy/run-001".to_owned()),
        artifact_sha256: None,
        independence_group: Some("model-fit-run-001".to_owned()),
    }
}

#[test]
fn model_inadequacy_requires_a_derived_adequacy_diagnostic() {
    let mut unsupported = profile();
    unsupported.findings.push(finding(
        Property::Learning,
        PropertyAssessment::ModelInadequate,
        &[],
    ));
    let errors = unsupported.validate().expect_err("diagnostic is mandatory");
    assert!(errors
        .iter()
        .any(|error| error.contains("requires a derived model-adequacy diagnostic")));

    let mut supported = profile();
    supported.evidence.push(model_diagnostic("residuals-001"));
    supported.findings.push(finding(
        Property::Learning,
        PropertyAssessment::ModelInadequate,
        &["residuals-001"],
    ));
    assert!(supported.validate().is_ok());
}

#[test]
fn temporal_undersampling_requires_an_explicit_aliasing_risk() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    let mut property = finding(
        Property::SelfMaintenance,
        PropertyAssessment::InsufficientEvidence,
        &["m1"],
    );
    let mut measured = coverage();
    measured.false_negative_risks.retain(|risk| *risk != FalseNegativeRisk::TemporalAliasing);
    property.coverage = Some(measured);
    candidate.findings.push(property);

    let errors = candidate.validate().expect_err("undersampling risk must be explicit");
    assert!(errors.iter().any(|error| error.contains("must record TemporalAliasing")));
}

#[test]
fn no_detection_requires_at_least_one_explicit_false_negative_risk() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    let mut property = finding(
        Property::SelfMaintenance,
        PropertyAssessment::InsufficientEvidence,
        &["m1"],
    );
    let mut measured = coverage();
    measured.false_negative_risks.clear();
    property.coverage = Some(measured);
    candidate.findings.push(property);

    let errors = candidate.validate().expect_err("false-negative risk is mandatory");
    assert!(errors.iter().any(|error| error.contains("false-negative risk mode")));
}

#[test]
fn calibrated_detection_probability_requires_a_linked_reference_record() {
    let mut candidate = profile();
    candidate.evidence.push(measurement("m1", EvidenceOutcome::NoDetection));
    let mut property = finding(
        Property::SelfMaintenance,
        PropertyAssessment::InsufficientEvidence,
        &["m1"],
    );
    let mut measured = coverage();
    measured.reference_abundance = Some(0.5);
    measured.detection_probability_at_reference = Some(0.95);
    measured.calibration_evidence_id = Some("calibration-001".to_owned());
    property.coverage = Some(measured);
    candidate.findings.push(property);

    let errors = candidate.validate().expect_err("calibration evidence must be linked");
    assert!(errors.iter().any(|error| error.contains("calibration_evidence_id references missing")));

    candidate.evidence.push(measurement("calibration-001", EvidenceOutcome::SignalDetected));
    assert!(candidate.validate().is_ok());
}

#[test]
fn detection_limit_requires_units() {
    let mut measured = coverage();
    measured.detection_limit_units = None;
    let mut issues = Vec::new();
    measured.validate("coverage", &mut issues);
    assert!(issues.iter().any(|error| error.contains("detection_limit requires non-empty units")));
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
