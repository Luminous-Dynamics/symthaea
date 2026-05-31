// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Symthaea Integrity Zome
//!
//! Defines entry types for HDC (Hyperdimensional Computing) operations
//! and AI-assisted design generation.

use hdi::prelude::*;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    HdcIntent(HdcIntentEntry),
    #[entry_type(visibility = "public")]
    GeneratedDesign(GeneratedDesignEntry),
    #[entry_type(visibility = "public")]
    SemanticMatch(SemanticMatchEntry),
    #[entry_type(visibility = "public")]
    OptimizationResult(OptimizationResultEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    IntentToDesigns,
    DesignToOptimizations,
    AuthorToIntents,
    SemanticSimilarity,
}

/// HDC Intent - A semantic query encoded as a hypervector
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HdcIntentEntry {
    pub description: String,
    pub vector_dimensions: u32,
    pub vector_hash: String, // Hash of the actual vector (stored externally)
    pub semantic_bindings: Vec<SerializedBinding>,
    pub generation_method: String,
    pub language: String,
    pub author: AgentPubKey,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SerializedBinding {
    pub concept: String,
    pub role: String, // Base, Modifier, Dimensional, Material, Functional
    pub weight: f32,
}

/// Generated Design - A parametric design created from intent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GeneratedDesignEntry {
    pub intent_hash: ActionHash,
    pub base_design_hash: Option<ActionHash>,
    pub parametric_config: String, // JSON of parameter values
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>, // JSON of printer requirements
    pub generated_file_cid: Option<String>,  // IPFS CID of generated file
    pub confidence_score: f32,
    pub generation_time_ms: u32,
    pub created_at: Timestamp,
}

/// Semantic Match - Result of similarity search
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SemanticMatchEntry {
    pub query_intent_hash: ActionHash,
    pub matched_design_hash: ActionHash,
    pub similarity_score: f32, // Cosine similarity in HDC space
    pub matched_bindings: Vec<String>,
    pub searched_at: Timestamp,
}

/// Optimization Result - Design optimized for local conditions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct OptimizationResultEntry {
    pub original_design_hash: ActionHash,
    pub optimized_for: OptimizationTarget,
    pub local_materials: Vec<ActionHash>,
    pub local_printers: Vec<ActionHash>,
    pub energy_preference: String,
    pub parameter_adjustments: String, // JSON of adjusted parameters
    pub improvement_metrics: String,   // JSON of improvement scores
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OptimizationTarget {
    MaterialAvailability,
    PrinterCapability,
    EnergyEfficiency,
    CostReduction,
    QualityMaximization,
    SpeedOptimization,
    Combined,
}

#[hdk_extern]
pub fn genesis_self_check(_: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::HdcIntent(intent) => {
                    if intent.description.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Description required".into(),
                        ));
                    }
                    if intent.vector_dimensions == 0 {
                        return Ok(ValidateCallbackResult::Invalid("Invalid dimensions".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::GeneratedDesign(design) => {
                    if design.confidence_score < 0.0 || design.confidence_score > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("Invalid confidence".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::SemanticMatch(m) => {
                    if m.similarity_score < 0.0 || m.similarity_score > 1.0 {
                        return Ok(ValidateCallbackResult::Invalid("Invalid similarity".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
