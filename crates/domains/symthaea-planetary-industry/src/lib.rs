// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neutral, evidence-bearing ontology for planetary industrial ecology.
//!
//! PIE-000 models what an industrial process graph contains. PIE-001 adds
//! interval-aware bulk-mass conservation. Chemistry, detailed thermodynamics,
//! equipment reproduction, optimization, and control authority belong to later
//! layers.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod graph;
mod mass_balance;
mod process;
mod types;

pub use graph::*;
pub use mass_balance::*;
pub use process::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn hypothesis(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            class: EvidenceClass::Hypothesis,
            source: String::new(),
            note: Some("synthetic test fixture only".into()),
        }
    }

    fn grade(label: &str) -> MaterialGrade {
        MaterialGrade {
            label: label.into(),
            specification_ref: None,
        }
    }

    fn synthetic_process(id: &str, disposition: OutputDisposition) -> ProcessDefinition {
        ProcessDefinition {
            process_id: id.into(),
            name: format!("Synthetic process {id}"),
            inputs: vec![ProcessInput {
                material_key: "synthetic-feed".into(),
                required_grade: Some(grade("research-grade")),
                role: ProcessInputRole::Feedstock,
                mass_kg: MassRangeKg::new(9.0, 11.0).unwrap(),
            }],
            outputs: vec![ProcessOutput {
                material_key: "synthetic-product".into(),
                grade: grade("unqualified-product"),
                form: PhysicalForm::Bulk,
                role: ProcessOutputRole::Product,
                mass_kg: MassRangeKg::new(5.0, 10.0).unwrap(),
                disposition,
            }],
            utilities: vec![
                UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(100.0, 200.0).unwrap()),
                UtilityDemand::PeakElectricalPower(PowerRangeW::new(10.0, 20.0).unwrap()),
            ],
            equipment: vec![EquipmentRequirement {
                equipment_class: "synthetic-reactor".into(),
                quantity: 1,
                criticality: DependencyCriticality::Essential,
                evidence: vec![hypothesis("equip-hyp")],
            }],
            environment: vec![EnvironmentConstraint::Body(CelestialBody::Moon)],
            evidence: vec![hypothesis("process-hyp")],
        }
    }

    #[test]
    fn ranges_fail_closed_on_negative_nonfinite_and_reversed_values() {
        assert!(MassRangeKg::new(-1.0, 2.0).is_err());
        assert!(MassRangeKg::new(3.0, 2.0).is_err());
        assert!(EnergyRangeJ::new(0.0, f64::NAN).is_err());
    }

    #[test]
    fn scalar_constructors_validate_and_expose_values() {
        let mass = MassKg::new(3.5).unwrap();
        assert_eq!(mass.value(), 3.5);
        assert!(MassKg::new(-0.1).is_err());
        assert!(MassKg::new(f64::INFINITY).is_err());
    }

    #[test]
    fn measured_evidence_requires_a_source() {
        let evidence = EvidenceRef {
            evidence_id: "lab-1".into(),
            class: EvidenceClass::LabMeasured,
            source: String::new(),
            note: None,
        };
        assert_eq!(evidence.validate(), Err(OntologyError::MissingEvidenceSource));
    }

    #[test]
    fn occurrence_is_not_a_material_lot() {
        let occurrence = ResourceOccurrence {
            resource_id: "occurrence-1".into(),
            name: "Synthetic icy regolith".into(),
            body: CelestialBody::Moon,
            state: ResourceState::Ice,
            in_place_mass_kg: Some(MassRangeKg::new(100.0, 200.0).unwrap()),
            evidence: vec![hypothesis("resource-hyp")],
        };
        let lot = MaterialLot {
            lot_id: "lot-1".into(),
            material_key: "processed-water".into(),
            grade: grade("industrial"),
            form: PhysicalForm::Liquid,
            mass_kg: MassKg::new(5.0).unwrap(),
            origin: MaterialOrigin {
                kind: MaterialOriginKind::ProcessedLocal,
                source_id: "batch-1".into(),
            },
            evidence: vec![hypothesis("lot-hyp")],
        };
        assert!(occurrence.validate().is_ok());
        assert!(lot.validate().is_ok());
        assert_ne!(occurrence.resource_id, lot.lot_id);
    }

    #[test]
    fn explicit_unknown_output_is_valid_but_not_hidden() {
        let process = synthetic_process("p1", OutputDisposition::Unknown);
        assert!(process.validate().is_ok());
        assert_eq!(process.outputs[0].disposition, OutputDisposition::Unknown);
    }

    #[test]
    fn zero_equipment_count_is_rejected() {
        let mut process = synthetic_process("p1", OutputDisposition::Inventory);
        process.equipment[0].quantity = 0;
        assert_eq!(process.validate(), Err(OntologyError::ZeroEquipmentCount));
    }

    #[test]
    fn graph_rejects_dangling_output_process_disposition() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![synthetic_process(
                "p1",
                OutputDisposition::Process("missing".into()),
            )],
            recycle_edges: vec![],
        };
        assert_eq!(
            graph.validate_structure(),
            Err(OntologyError::UnknownProcess("missing".into()))
        );
    }

    #[test]
    fn graph_rejects_dangling_recycle_edges() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![synthetic_process("p1", OutputDisposition::Inventory)],
            recycle_edges: vec![RecycleEdge {
                edge_id: "r1".into(),
                from_process_id: "p1".into(),
                material_key: "synthetic-product".into(),
                to_process_id: "missing".into(),
                recovered_mass_kg: MassRangeKg::new(1.0, 2.0).unwrap(),
                evidence: vec![hypothesis("recycle-hyp")],
            }],
        };
        assert_eq!(
            graph.validate_structure(),
            Err(OntologyError::UnknownProcess("missing".into()))
        );
    }

    #[test]
    fn graph_accepts_structurally_closed_process_and_recycle_edges() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![
                synthetic_process("p1", OutputDisposition::Process("p2".into())),
                synthetic_process("p2", OutputDisposition::Inventory),
            ],
            recycle_edges: vec![RecycleEdge {
                edge_id: "r1".into(),
                from_process_id: "p1".into(),
                material_key: "synthetic-product".into(),
                to_process_id: "p2".into(),
                recovered_mass_kg: MassRangeKg::new(1.0, 2.0).unwrap(),
                evidence: vec![hypothesis("recycle-hyp")],
            }],
        };
        assert!(graph.validate_structure().is_ok());
    }

    #[test]
    fn duplicate_ids_fail_closed_within_namespace() {
        let graph = IndustrialProcessGraph {
            resources: vec![],
            lots: vec![],
            processes: vec![
                synthetic_process("dup", OutputDisposition::Inventory),
                synthetic_process("dup", OutputDisposition::Inventory),
            ],
            recycle_edges: vec![],
        };
        assert_eq!(
            graph.validate_structure(),
            Err(OntologyError::DuplicateId("dup".into()))
        );
    }

    #[test]
    fn matter_has_no_utility_side_channel() {
        let utilities = [
            UtilityDemand::ElectricalEnergy(EnergyRangeJ::new(1.0, 2.0).unwrap()),
            UtilityDemand::ThermalEnergy(EnergyRangeJ::new(1.0, 2.0).unwrap()),
            UtilityDemand::CoolingEnergy(EnergyRangeJ::new(1.0, 2.0).unwrap()),
            UtilityDemand::PeakElectricalPower(PowerRangeW::new(1.0, 2.0).unwrap()),
            UtilityDemand::ProcessTime(DurationRangeS::new(1.0, 2.0).unwrap()),
        ];
        assert_eq!(utilities.len(), 5);
    }
}
