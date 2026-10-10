//! Minimal passive-design benchmark.
//!
//! Exercises the conservative path without claiming that the cube is a useful
//! engineering design. Its purpose is to keep the passive-objective plumbing
//! executable and regression-testable.

use symthaea_fabrication_kernel::{
    CSGNode, PassiveDesignEvidence, PassiveEvidenceFact, PassiveEvidenceExtractor,
    PassiveEvidenceSource, PassiveFunctionContract, PassiveInput, PassiveMechanism, PassiveOutput,
    PassiveObjectiveWeights,
};

fn complete_zero_actuation_facts(fluid_motion: bool) -> [PassiveEvidenceFact; 6] {
    [
        PassiveEvidenceFact::MovingSolidComponents {
            count: 0,
            source: PassiveEvidenceSource::DesignDeclaration,
        },
        PassiveEvidenceFact::MechanicalJoints {
            count: 0,
            source: PassiveEvidenceSource::DesignDeclaration,
        },
        PassiveEvidenceFact::ActivePowerWatts {
            watts: 0.0,
            source: PassiveEvidenceSource::SimulationDeclaration,
        },
        PassiveEvidenceFact::CommandedActuators {
            count: 0,
            source: PassiveEvidenceSource::DesignDeclaration,
        },
        PassiveEvidenceFact::ExternalControlRequired {
            required: false,
            source: PassiveEvidenceSource::DesignDeclaration,
        },
        PassiveEvidenceFact::FluidMotionUsed {
            used: fluid_motion,
            source: PassiveEvidenceSource::SimulationDeclaration,
        },
    ]
}

fn main() {
    let contract = PassiveFunctionContract::strict(
        PassiveInput::Fluidic,
        PassiveOutput::Fluidic,
        PassiveMechanism::Geometry,
    );

    let passive_facts = complete_zero_actuation_facts(true);
    let extraction = PassiveEvidenceExtractor::extract(&passive_facts);
    assert!(
        extraction.conflicts.is_empty(),
        "benchmark fixture must be internally consistent"
    );
    assert!(
        extraction.missing_fields.is_empty(),
        "benchmark fixture must cover all strict passive fields"
    );

    let mesh = CSGNode::cube();
    let (passive_obs, passive_report) =
        symthaea_fabrication_kernel::generative::passive_objective_observation(
            &mesh,
            12,
            100.0,
            &contract,
            extraction.evidence,
        )
        .expect("cube must be evaluable by the existing analytical path");

    let passive_score = passive_obs.weighted_score(PassiveObjectiveWeights::default());
    println!("passive compliance: {}", passive_report.compliant);
    println!("passive score: {:.6}", passive_obs.passivity);
    println!("combined objective: {:.6}", passive_score);

    let active_evidence = PassiveDesignEvidence {
        commanded_actuators: 1,
        ..extraction.evidence
    };
    let active_report = contract.validate(active_evidence);
    assert!(
        !active_report.compliant,
        "an active actuator must violate the strict passive contract"
    );
    println!(
        "active-control regression: rejected ({:?})",
        active_report.violations
    );
}
