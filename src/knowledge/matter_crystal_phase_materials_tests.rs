// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use crate::knowledge::matter_cross_scale::MatterScaleTransition;
use crate::knowledge::matter_cross_scale_lineage::CrossScaleLineageEnvelope;
use crate::knowledge::matter_crystal_phase::{
    admit_crystal_phase_candidate, classify_current_compound_stability_screening,
    CrystalPhaseAdmissionError, CrystalPhaseEvidence, CrystalPhaseTarget,
};
use symthaea_epistemic_types::{
    MatterClaim, MatterScale, MatterSolverCapability, MatterSolverProfile, MatterValidationStage,
};

fn solver(name: &str, capabilities: Vec<MatterSolverCapability>) -> MatterSolverProfile {
    MatterSolverProfile {
        name: name.to_string(),
        version: "1".to_string(),
        method: format!("{name} fixture"),
        capabilities,
        limitations: vec!["fixture only".to_string()],
    }
}

fn claim(
    id: &str,
    scale: MatterScale,
    capabilities: Vec<MatterSolverCapability>,
) -> MatterClaim {
    MatterClaim::local_computational(
        id,
        format!("statement for {id}"),
        scale,
        MatterValidationStage::HighFidelitySimulated,
        solver(id, capabilities),
    )
    .unwrap()
}

fn admitted_crystal_candidate() -> crate::knowledge::matter_cross_scale_lineage::AdmittedCrossScaleMatterClaim {
    let source = claim("electronic-source", MatterScale::AtomicElectronic, vec![]);
    let transition_evidence = claim("electronic-to-crystal", MatterScale::Crystal, vec![]);
    let transition = MatterScaleTransition::supported_in_domain(
        "electronic->crystal",
        MatterScale::AtomicElectronic,
        MatterScale::Crystal,
        &transition_evidence,
    );
    let envelope = CrossScaleLineageEnvelope::assess(
        &[&source],
        &[transition],
        MatterScale::Crystal,
    )
    .unwrap();
    envelope
        .admit_derived_computational_claim(claim("crystal-candidate", MatterScale::Crystal, vec![]))
        .unwrap()
}

#[test]
fn real_legacy_stability_true_still_cannot_replace_phase_competition() {
    let prediction = symthaea_materials::compound_stability::predict_stability(
        &[(11, 0.5), (17, 0.5)],
        300.0,
    );
    assert!(prediction.is_stable);

    let advisory = classify_current_compound_stability_screening(&prediction);
    assert!(advisory.reported_stability_flag);

    let periodic = claim(
        "periodic",
        MatterScale::Crystal,
        vec![MatterSolverCapability::PeriodicElectronicStructure],
    );
    let relaxation = claim("relaxation", MatterScale::Crystal, vec![]);
    let dynamic = claim("dynamic", MatterScale::Crystal, vec![]);
    let heuristic_claim = MatterClaim::local_computational(
        "legacy-composition-screen",
        format!(
            "legacy advisory screen for {}: proxy={} reported_flag={}",
            advisory.formula,
            advisory.formation_energy_proxy_ev_per_atom,
            advisory.reported_stability_flag
        ),
        MatterScale::Crystal,
        MatterValidationStage::SurrogateSupported,
        solver("legacy-composition-screen", vec![]),
    )
    .unwrap();

    let evidence = vec![
        CrystalPhaseEvidence::periodic_electronic_structure(&periodic).unwrap(),
        CrystalPhaseEvidence::structural_relaxation(
            &relaxation,
            0.01,
            0.02,
            "criterion:relax:v1",
        )
        .unwrap(),
        CrystalPhaseEvidence::dynamical_stability(
            &dynamic,
            0.1,
            0.02,
            64,
            "criterion:phonon:v1",
        )
        .unwrap(),
        CrystalPhaseEvidence::heuristic_composition_screening(&heuristic_claim).unwrap(),
    ];

    let target = CrystalPhaseTarget::try_new(vec![11, 17], None).unwrap();
    assert_eq!(
        admit_crystal_phase_candidate(
            admitted_crystal_candidate(),
            target,
            &evidence,
            None,
            None,
        )
        .unwrap_err(),
        CrystalPhaseAdmissionError::HeuristicScreeningCannotEstablishPhaseStability
    );
}
