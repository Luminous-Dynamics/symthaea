// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Graph-level structural closure for Planetary Industrial Ecology.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::process::{OutputDisposition, ProcessDefinition};
use crate::types::{
    EvidenceRef, MassRangeKg, MaterialLot, OntologyError, ResourceOccurrence, require_text,
    validate_evidence,
};

/// Explicit recycle/reprocessing edge between two known processes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecycleEdge {
    /// Stable edge identifier.
    pub edge_id: String,
    /// Process producing the recyclable material.
    pub from_process_id: String,
    /// Material identity routed on this edge.
    pub material_key: String,
    /// Process consuming/reprocessing the material.
    pub to_process_id: String,
    /// Maximum recovered material routed on the declared basis.
    pub recovered_mass_kg: MassRangeKg,
    /// Supporting evidence or explicit hypothesis.
    pub evidence: Vec<EvidenceRef>,
}

impl RecycleEdge {
    fn validate(&self) -> Result<(), OntologyError> {
        require_text(&self.edge_id, "recycle_edge_id")?;
        require_text(&self.from_process_id, "recycle_from_process_id")?;
        require_text(&self.to_process_id, "recycle_to_process_id")?;
        require_text(&self.material_key, "recycle_material_key")?;
        self.recovered_mass_kg.validate()?;
        validate_evidence(&self.evidence)
    }
}

/// Structurally closed PIE process graph.
///
/// This validates IDs and process references only; it does not claim that
/// material/energy balances close or that resources are geographically available.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndustrialProcessGraph {
    /// Resource occurrences considered by the study.
    pub resources: Vec<ResourceOccurrence>,
    /// Existing inventory lots considered by the study.
    pub lots: Vec<MaterialLot>,
    /// Candidate industrial processes.
    pub processes: Vec<ProcessDefinition>,
    /// Explicit recycling/reprocessing links.
    pub recycle_edges: Vec<RecycleEdge>,
}

impl IndustrialProcessGraph {
    /// Validate record integrity, namespace uniqueness, and process-reference closure.
    pub fn validate_structure(&self) -> Result<(), OntologyError> {
        let mut resource_ids = BTreeSet::new();
        for resource in &self.resources {
            resource.validate()?;
            if !resource_ids.insert(resource.resource_id.clone()) {
                return Err(OntologyError::DuplicateId(resource.resource_id.clone()));
            }
        }

        let mut lot_ids = BTreeSet::new();
        for lot in &self.lots {
            lot.validate()?;
            if !lot_ids.insert(lot.lot_id.clone()) {
                return Err(OntologyError::DuplicateId(lot.lot_id.clone()));
            }
        }

        let mut process_ids = BTreeSet::new();
        for process in &self.processes {
            process.validate()?;
            if !process_ids.insert(process.process_id.clone()) {
                return Err(OntologyError::DuplicateId(process.process_id.clone()));
            }
        }

        // A declared downstream process disposition is a graph edge and must
        // resolve just like an explicit recycle edge. Unknown destinations stay
        // explicit through `OutputDisposition::Unknown`; dangling IDs are errors.
        for process in &self.processes {
            for output in &process.outputs {
                if let OutputDisposition::Process(target) = &output.disposition {
                    if !process_ids.contains(target) {
                        return Err(OntologyError::UnknownProcess(target.clone()));
                    }
                }
            }
        }

        let mut edge_ids = BTreeSet::new();
        for edge in &self.recycle_edges {
            edge.validate()?;
            if !edge_ids.insert(edge.edge_id.clone()) {
                return Err(OntologyError::DuplicateId(edge.edge_id.clone()));
            }
            if !process_ids.contains(&edge.from_process_id) {
                return Err(OntologyError::UnknownProcess(edge.from_process_id.clone()));
            }
            if !process_ids.contains(&edge.to_process_id) {
                return Err(OntologyError::UnknownProcess(edge.to_process_id.clone()));
            }
        }
        Ok(())
    }
}
