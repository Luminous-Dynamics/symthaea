// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_science_evidence::{
    MeasurementNoiseModel, ModelPredictionComparison, NoiseCalibrationStatus,
    OTHER_UNRESOLVED_MODEL_ID, ObservableContract, PredictionDisposition,
    THEORY_PREDICTION_SCHEMA, TheoryModel, TheoryModelRole, TheoryPredictionLedger,
    TheoryPredictionRecord,
};

const SOURCE_REVISION: &str = "f90039dfc043a55b49e891ab1627ac1b6dd58f75";

fn model(id: &str, role: TheoryModelRole) -> TheoryModel {
    TheoryModel {
        theory_id: id.to_string(),
        family: id.to_string(),
        model_spec_blake3_hex: blake3::hash(id.as_bytes()).to_hex().to_string(),
        source_revision_sha: SOURCE_REVISION.to_string(),
        role,
    }
}

fn prediction(
    theory_id: &str,
    observable_id: &str,
    disposition: PredictionDisposition,
    scope: &str,
    units: &str,
    value: Option<f64>,
    derivation: Option<&str>,
    notes: &str,
) -> TheoryPredictionRecord {
    TheoryPredictionRecord {
        theory_id: theory_id.to_string(),
        observable_id: observable_id.to_string(),
        disposition,
        scope_id: scope.to_string(),
        units: units.to_string(),
        predicted_value: value,
        derivation_reference: derivation.map(str::to_string),
        notes: notes.to_string(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let models = vec![
        model("causal-set-minkowski", TheoryModelRole::NamedTheory),
        model("minkowski-continuum", TheoryModelRole::NamedTheory),
        model("chain-theory", TheoryModelRole::NamedTheory),
        model(OTHER_UNRESOLVED_MODEL_ID, TheoryModelRole::OtherUnresolved),
    ];
    let observables = vec![
        ObservableContract {
            observable_id: "ordering-fraction-1p1".to_string(),
            units: "fraction".to_string(),
            operational_definition: "fraction of unordered event pairs that are causally related".to_string(),
            domain_id: "poisson-sprinkling-minkowski-1p1".to_string(),
            noise_model: MeasurementNoiseModel::Gaussian,
            measurement_sigma: Some(0.01),
            noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
            calibration_reference: Some("synthetic-known-answer-noise-v1".to_string()),
        },
        ObservableContract {
            observable_id: "inverse-alpha-candidate".to_string(),
            units: "dimensionless".to_string(),
            operational_definition: "candidate inverse fine-structure readout under the paper's construction".to_string(),
            domain_id: "low-energy-fine-structure-constant-definition".to_string(),
            noise_model: MeasurementNoiseModel::Unspecified,
            measurement_sigma: None,
            noise_calibration: NoiseCalibrationStatus::Uncalibrated,
            calibration_reference: None,
        },
        ObservableContract {
            observable_id: "photon-propagator".to_string(),
            units: "declared-amplitude-units".to_string(),
            operational_definition: "momentum-dependent propagator under a fixed scattering protocol".to_string(),
            domain_id: "fixed-momentum-scattering-protocol".to_string(),
            noise_model: MeasurementNoiseModel::Unspecified,
            measurement_sigma: None,
            noise_calibration: NoiseCalibrationStatus::Uncalibrated,
            calibration_reference: None,
        },
    ];
    let predictions = vec![
        prediction("causal-set-minkowski","ordering-fraction-1p1",PredictionDisposition::Quantitative,"poisson-sprinkling-minkowski-1p1","fraction",Some(0.5),Some("Myrheim-Meyer expected ordering fraction, d=2"),"Analytic reference."),
        prediction("minkowski-continuum","ordering-fraction-1p1",PredictionDisposition::Quantitative,"poisson-sprinkling-minkowski-1p1","fraction",Some(0.5),Some("continuum Alexandrov-interval ordering reference"),"Same observable prediction; this fixture does not discriminate the ontologies."),
        prediction("chain-theory","ordering-fraction-1p1",PredictionDisposition::NotDerived,"qudit-entanglement-network","fraction",None,None,"No quantitative causal-order prediction is registered."),
        prediction(OTHER_UNRESOLVED_MODEL_ID,"ordering-fraction-1p1",PredictionDisposition::Unresolved,"open-world","fraction",None,None,"Unknown-state predictive envelope not calibrated."),
        prediction("chain-theory","inverse-alpha-candidate",PredictionDisposition::Quantitative,"low-energy-fine-structure-constant-definition","dimensionless",Some(137.035_999_189_566),Some("published finite-dimensional record/readout construction"),"Candidate arithmetic value only; not a propagator prediction."),
        prediction("causal-set-minkowski","inverse-alpha-candidate",PredictionDisposition::NotDerived,"causal-set-minkowski","dimensionless",None,None,"Not derived."),
        prediction("minkowski-continuum","inverse-alpha-candidate",PredictionDisposition::NotDerived,"standard-model-parameter-input","dimensionless",None,None,"The conventional parameterization takes the coupling as an empirical input."),
        prediction(OTHER_UNRESOLVED_MODEL_ID,"inverse-alpha-candidate",PredictionDisposition::Unresolved,"open-world","dimensionless",None,None,"Unknown-state predictive envelope not calibrated."),
        prediction("chain-theory","photon-propagator",PredictionDisposition::NotDerived,"qudit-entanglement-network","declared-amplitude-units",None,None,"Published construction does not yet supply the momentum-dependent propagator."),
        prediction("causal-set-minkowski","photon-propagator",PredictionDisposition::OutsideValidatedDomain,"causal-set-link-propagator-not-implemented","declared-amplitude-units",None,None,"No validated propagator implementation admitted."),
        prediction("minkowski-continuum","photon-propagator",PredictionDisposition::QualitativeOnly,"fixed-momentum-scattering-protocol","declared-amplitude-units",None,Some("QED propagator formalism; protocol not pinned"),"No fixed kinematic protocol or likelihood supplied."),
        prediction(OTHER_UNRESOLVED_MODEL_ID,"photon-propagator",PredictionDisposition::Unresolved,"open-world","declared-amplitude-units",None,None,"Unknown-state predictive envelope not calibrated."),
    ];

    let ledger = TheoryPredictionLedger {
        schema: THEORY_PREDICTION_SCHEMA.to_string(),
        models,
        observables,
        predictions,
    };
    ledger.validate()?;
    let coverage = ledger.coverage()?;

    println!("epistemic_status=SYNTHETIC_THEORY_PREDICTION_COVERAGE_NOT_PHYSICAL_MODEL_VALIDATION");
    println!(
        "coverage,total={},quantitative={},qualitative={},not_derived={},outside_domain={},unresolved={},quantitative_fraction={:.6}",
        coverage.total_model_observable_pairs,
        coverage.quantitative,
        coverage.qualitative_only,
        coverage.not_derived,
        coverage.outside_validated_domain,
        coverage.unresolved,
        coverage.quantitative_fraction(),
    );

    match ledger.compare_models("causal-set-minkowski","minkowski-continuum","ordering-fraction-1p1")? {
        ModelPredictionComparison::Comparable { absolute_difference, measurement_sigma, normalized_separation, .. } =>
            println!("causal_set_vs_continuum=COMPARABLE,delta={absolute_difference:.6},sigma={measurement_sigma:.6},separation_sigma={normalized_separation:.6}"),
        ModelPredictionComparison::NotComparable { blockers, .. } =>
            println!("causal_set_vs_continuum=NOT_COMPARABLE,blockers={blockers:?}"),
    }
    match ledger.compare_models("chain-theory","minkowski-continuum","photon-propagator")? {
        ModelPredictionComparison::Comparable { normalized_separation, .. } =>
            println!("chain_theory_vs_qed_propagator=COMPARABLE,separation_sigma={normalized_separation:.6}"),
        ModelPredictionComparison::NotComparable { blockers, .. } =>
            println!("chain_theory_vs_qed_propagator=NOT_COMPARABLE,blockers={blockers:?}"),
    }
    println!("claim_boundary=prediction_coverage_and_pairwise_comparability_only");
    println!("not_claimed=physical_theory_selection,chain_theory_propagator,UAP_origin,new_physics");
    Ok(())
}
