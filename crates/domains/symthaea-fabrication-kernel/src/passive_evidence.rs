// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! High-confidence extraction of passive-design evidence.
//!
//! The extractor deliberately consumes structured declarations rather than
//! guessing kinematics from arbitrary triangle meshes. This makes the output
//! suitable for evidence pipelines: every field has an explicit source and
//! ambiguous CAD remains "unknown" instead of becoming an accidental claim.

use crate::passive_design::{PassiveDesignEvidence, PassiveFunctionContract, PassiveValidationReport};

/// Source of a structured passive-design fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassiveEvidenceSource {
    /// Declared directly by the design/assembly model.
    DesignDeclaration,
    /// Declared by a simulation/control model.
    SimulationDeclaration,
    /// Declared by a manufacturing/process specification.
    ManufacturingDeclaration,
}

/// One structured fact supplied to the passive evidence extractor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PassiveEvidenceFact {
    MovingSolidComponents {
        count: usize,
        source: PassiveEvidenceSource,
    },
    MechanicalJoints {
        count: usize,
        source: PassiveEvidenceSource,
    },
    ActivePowerWatts {
        watts: f64,
        source: PassiveEvidenceSource,
    },
    CommandedActuators {
        count: usize,
        source: PassiveEvidenceSource,
    },
    ExternalControlRequired {
        required: bool,
        source: PassiveEvidenceSource,
    },
    FluidMotionUsed {
        used: bool,
        source: PassiveEvidenceSource,
    },
    DistributedDeformationUsed {
        used: bool,
        source: PassiveEvidenceSource,
    },
    PhaseChangeUsed {
        used: bool,
        source: PassiveEvidenceSource,
    },
}

/// Result of extracting evidence from structured facts.
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveEvidenceExtraction {
    pub evidence: PassiveDesignEvidence,
    pub sources: Vec<PassiveEvidenceSource>,
    pub conflicts: Vec<PassiveEvidenceConflict>,
}

/// A fact was supplied more than once with contradictory values.
#[derive(Debug, Clone, PartialEq)]
pub enum PassiveEvidenceConflict {
    MovingSolidComponents { first: usize, second: usize },
    MechanicalJoints { first: usize, second: usize },
    ActivePowerWatts { first: f64, second: f64 },
    CommandedActuators { first: usize, second: usize },
    ExternalControlRequired { first: bool, second: bool },
    FluidMotionUsed { first: bool, second: bool },
    DistributedDeformationUsed { first: bool, second: bool },
    PhaseChangeUsed { first: bool, second: bool },
}

/// Conservative extractor for high-confidence structured evidence.
#[derive(Debug, Default)]
pub struct PassiveEvidenceExtractor;

impl PassiveEvidenceExtractor {
    /// Extract evidence without inferring missing facts.
    ///
    /// Duplicate facts must agree. A conflict is retained rather than
    /// resolved by precedence, because silently choosing one declaration
    /// would hide an epistemic inconsistency.
    pub fn extract(facts: &[PassiveEvidenceFact]) -> PassiveEvidenceExtraction {
        let mut evidence = PassiveDesignEvidence::default();
        let mut sources = Vec::new();
        let mut conflicts = Vec::new();

        let mut moving = None;
        let mut joints = None;
        let mut active_power = None;
        let mut actuators = None;
        let mut external_control = None;
        let mut fluid_motion = None;
        let mut distributed_deformation = None;
        let mut phase_change = None;

        for fact in facts {
            let source = match fact {
                PassiveEvidenceFact::MovingSolidComponents { source, .. }
                | PassiveEvidenceFact::MechanicalJoints { source, .. }
                | PassiveEvidenceFact::ActivePowerWatts { source, .. }
                | PassiveEvidenceFact::CommandedActuators { source, .. }
                | PassiveEvidenceFact::ExternalControlRequired { source, .. }
                | PassiveEvidenceFact::FluidMotionUsed { source, .. }
                | PassiveEvidenceFact::DistributedDeformationUsed { source, .. }
                | PassiveEvidenceFact::PhaseChangeUsed { source, .. } => *source,
            };
            sources.push(source);

            match *fact {
                PassiveEvidenceFact::MovingSolidComponents { count, .. } => {
                    if let Some(first) = moving {
                        if first != count {
                            conflicts.push(PassiveEvidenceConflict::MovingSolidComponents {
                                first,
                                second: count,
                            });
                        }
                    } else {
                        moving = Some(count);
                        evidence.moving_solid_components = count;
                    }
                }
                PassiveEvidenceFact::MechanicalJoints { count, .. } => {
                    if let Some(first) = joints {
                        if first != count {
                            conflicts.push(PassiveEvidenceConflict::MechanicalJoints {
                                first,
                                second: count,
                            });
                        }
                    } else {
                        joints = Some(count);
                        evidence.mechanical_joints = count;
                    }
                }
                PassiveEvidenceFact::ActivePowerWatts { watts, .. } => {
                    if let Some(first) = active_power {
                        if first != watts {
                            conflicts.push(PassiveEvidenceConflict::ActivePowerWatts {
                                first,
                                second: watts,
                            });
                        }
                    } else {
                        active_power = Some(watts);
                        evidence.active_power_w = watts;
                    }
                }
                PassiveEvidenceFact::CommandedActuators { count, .. } => {
                    if let Some(first) = actuators {
                        if first != count {
                            conflicts.push(PassiveEvidenceConflict::CommandedActuators {
                                first,
                                second: count,
                            });
                        }
                    } else {
                        actuators = Some(count);
                        evidence.commanded_actuators = count;
                    }
                }
                PassiveEvidenceFact::ExternalControlRequired { required, .. } => {
                    if let Some(first) = external_control {
                        if first != required {
                            conflicts.push(PassiveEvidenceConflict::ExternalControlRequired {
                                first,
                                second: required,
                            });
                        }
                    } else {
                        external_control = Some(required);
                        evidence.requires_external_control = required;
                    }
                }
                PassiveEvidenceFact::FluidMotionUsed { used, .. } => {
                    if let Some(first) = fluid_motion {
                        if first != used {
                            conflicts.push(PassiveEvidenceConflict::FluidMotionUsed {
                                first,
                                second: used,
                            });
                        }
                    } else {
                        fluid_motion = Some(used);
                        evidence.uses_fluid_motion = used;
                    }
                }
                PassiveEvidenceFact::DistributedDeformationUsed { used, .. } => {
                    if let Some(first) = distributed_deformation {
                        if first != used {
                            conflicts.push(
                                PassiveEvidenceConflict::DistributedDeformationUsed {
                                    first,
                                    second: used,
                                },
                            );
                        }
                    } else {
                        distributed_deformation = Some(used);
                        evidence.uses_distributed_deformation = used;
                    }
                }
                PassiveEvidenceFact::PhaseChangeUsed { used, .. } => {
                    if let Some(first) = phase_change {
                        if first != used {
                            conflicts.push(PassiveEvidenceConflict::PhaseChangeUsed {
                                first,
                                second: used,
                            });
                        }
                    } else {
                        phase_change = Some(used);
                        evidence.uses_phase_change = used;
                    }
                }
            }
        }

        sources.sort_unstable_by_key(|source| *source as u8);
        sources.dedup();

        PassiveEvidenceExtraction {
            evidence,
            sources,
            conflicts,
        }
    }

    /// Extract and validate against a contract.
    ///
    /// Any evidence conflict is treated as non-compliant. This prevents a
    /// generated candidate from selecting whichever declaration is favorable.
    pub fn validate(
        contract: &PassiveFunctionContract,
        facts: &[PassiveEvidenceFact],
    ) -> PassiveValidationReport {
        let extraction = Self::extract(facts);
        let mut report = contract.validate(extraction.evidence);

        if !extraction.conflicts.is_empty() {
            report.compliant = false;
            report
                .violations
                .extend(
                    std::iter::repeat_n(
                        crate::passive_design::PassiveViolation::EvidenceConflict,
                        extraction.conflicts.len(),
                    ),
                );
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passive_design::{PassiveInput, PassiveMechanism, PassiveOutput};

    #[test]
    fn empty_facts_are_explicitly_passive_but_not_physically_proven() {
        let extraction = PassiveEvidenceExtractor::extract(&[]);
        assert_eq!(extraction.evidence, PassiveDesignEvidence::default());
        assert!(extraction.sources.is_empty());
        assert!(extraction.conflicts.is_empty());
    }

    #[test]
    fn duplicate_agreeing_facts_are_deduplicated() {
        let facts = [
            PassiveEvidenceFact::MechanicalJoints {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::MechanicalJoints {
                count: 0,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let extraction = PassiveEvidenceExtractor::extract(&facts);
        assert_eq!(extraction.evidence.mechanical_joints, 0);
        assert!(extraction.conflicts.is_empty());
        assert_eq!(extraction.sources.len(), 2);
    }

    #[test]
    fn contradictory_facts_are_preserved_as_conflict() {
        let facts = [
            PassiveEvidenceFact::MovingSolidComponents {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::MovingSolidComponents {
                count: 2,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let extraction = PassiveEvidenceExtractor::extract(&facts);
        assert_eq!(extraction.conflicts.len(), 1);
        assert_eq!(
            extraction.evidence.moving_solid_components,
            0,
            "first observed value remains the evidence value"
        );
    }

    #[test]
    fn zero_and_nonzero_facts_conflict() {
        let facts = [
            PassiveEvidenceFact::MechanicalJoints {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::MechanicalJoints {
                count: 1,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let extraction = PassiveEvidenceExtractor::extract(&facts);
        assert_eq!(extraction.conflicts.len(), 1);
        assert!(matches!(
            extraction.conflicts[0],
            PassiveEvidenceConflict::MechanicalJoints { first: 0, second: 1 }
        ));
    }

    #[test]
    fn false_and_true_facts_conflict() {
        let facts = [
            PassiveEvidenceFact::ExternalControlRequired {
                required: false,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::ExternalControlRequired {
                required: true,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let extraction = PassiveEvidenceExtractor::extract(&facts);
        assert_eq!(extraction.conflicts.len(), 1);
    }

    #[test]
    fn active_power_is_rejected_by_strict_contract() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Electromagnetic,
            PassiveOutput::Optical,
            PassiveMechanism::FieldInteraction,
        );
        let facts = [PassiveEvidenceFact::ActivePowerWatts {
            watts: 2.0,
            source: PassiveEvidenceSource::SimulationDeclaration,
        }];
        let report = PassiveEvidenceExtractor::validate(&contract, &facts);
        assert!(!report.compliant);
    }

    #[test]
    fn fluidic_state_change_remains_allowed_with_complete_evidence() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Fluidic,
            PassiveOutput::Fluidic,
            PassiveMechanism::Geometry,
        );
        let facts = [
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
                used: true,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let report = PassiveEvidenceExtractor::validate(&contract, &facts);
        assert!(report.compliant);
    }

    #[test]
    fn contradiction_cannot_be_hidden_by_validation() {
        let contract = PassiveFunctionContract::strict(
            PassiveInput::Mechanical,
            PassiveOutput::Mechanical,
            PassiveMechanism::Geometry,
        );
        let facts = [
            PassiveEvidenceFact::MechanicalJoints {
                count: 0,
                source: PassiveEvidenceSource::DesignDeclaration,
            },
            PassiveEvidenceFact::MechanicalJoints {
                count: 1,
                source: PassiveEvidenceSource::SimulationDeclaration,
            },
        ];
        let report = PassiveEvidenceExtractor::validate(&contract, &facts);
        assert!(!report.compliant);
    }
}
