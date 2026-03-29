// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Integrity Zome
//!
//! Defines entry types for HDC (Hyperdimensional Computing) operations
//! and AI-assisted design generation.

use hdi::prelude::*;
use fabrication_common::check;
use fabrication_common::validation;

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
    AnchorToIntents,
    CategoryToIntents,
    SemanticSimilarity,
    RateLimitBucket,
}

/// HDC Intent - A semantic query encoded as a hypervector
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HdcIntentEntry {
    pub description: String,
    pub vector_dimensions: u32,
    pub vector_hash: String,  // Hash of the actual vector (stored externally)
    pub semantic_bindings: Vec<SerializedBinding>,
    pub generation_method: String,
    pub language: String,
    pub author: AgentPubKey,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SerializedBinding {
    pub concept: String,
    pub role: String,  // Base, Modifier, Dimensional, Material, Functional
    pub weight: f32,
}

/// Generated Design - A parametric design created from intent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GeneratedDesignEntry {
    pub intent_hash: ActionHash,
    pub base_design_hash: Option<ActionHash>,
    pub parametric_config: String,  // JSON of parameter values
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>,  // JSON of printer requirements
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
    pub similarity_score: f32,  // Cosine similarity in HDC space
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
    pub parameter_adjustments: String,  // JSON of adjusted parameters
    pub improvement_metrics: String,  // JSON of improvement scores
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

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry { app_entry, .. } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            let max_len: usize = 256;
            check!(validation::require_max_tag_len(&tag, max_len, &format!("{:?}", link_type)));
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            if action.author != *original_action.action().author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(op_update) => {
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Dispatch entry validation to per-type functions
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::HdcIntent(intent) => validate_hdc_intent(intent),
        EntryTypes::GeneratedDesign(design) => validate_generated_design(design),
        EntryTypes::SemanticMatch(m) => validate_semantic_match(m),
        EntryTypes::OptimizationResult(o) => validate_optimization_result(o),
    }
}

/// Validate an HdcIntentEntry
fn validate_hdc_intent(intent: HdcIntentEntry) -> ExternResult<ValidateCallbackResult> {
    // description: non-empty (trim), max 4096 chars
    check!(validation::require_non_empty(&intent.description, "description"));
    check!(validation::require_max_len(&intent.description, 4096, "description"));

    // vector_dimensions: must be a power of 2 in [4096, 16384]
    if !intent.vector_dimensions.is_power_of_two()
        || intent.vector_dimensions < 4096
        || intent.vector_dimensions > 16384
    {
        return Ok(ValidateCallbackResult::Invalid(
            "vector_dimensions must be a power of 2 between 4096 and 16384".to_string(),
        ));
    }

    // vector_hash: non-empty, max 256 chars
    check!(validation::require_non_empty(&intent.vector_hash, "vector_hash"));
    check!(validation::require_max_len(&intent.vector_hash, 256, "vector_hash"));

    // generation_method: max 64 chars
    check!(validation::require_max_len(&intent.generation_method, 64, "generation_method"));

    // language: non-empty, max 32 chars
    check!(validation::require_non_empty(&intent.language, "language"));
    check!(validation::require_max_len(&intent.language, 32, "language"));

    // semantic_bindings: max 128 items
    check!(validation::require_max_vec_len(&intent.semantic_bindings, 128, "semantic_bindings"));

    // Each binding: weight in range, concept max 256, role max 32
    for binding in &intent.semantic_bindings {
        check!(validation::require_in_range(binding.weight, 0.0, 1.0, "binding weight"));
        check!(validation::require_max_len(&binding.concept, 256, "binding concept"));
        check!(validation::require_max_len(&binding.role, 32, "binding role"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a GeneratedDesignEntry
fn validate_generated_design(design: GeneratedDesignEntry) -> ExternResult<ValidateCallbackResult> {
    // confidence_score: in range 0.0..1.0 (also catches NaN/Inf)
    check!(validation::require_in_range(design.confidence_score, 0.0, 1.0, "confidence_score"));

    // parametric_config: max 32768 chars
    check!(validation::require_max_len(&design.parametric_config, 32768, "parametric_config"));

    // material_constraints: max 64 items, each max 256 chars
    check!(validation::require_max_vec_len(&design.material_constraints, 64, "material_constraints"));
    for constraint in &design.material_constraints {
        check!(validation::require_max_len(constraint, 256, "material constraint"));
    }

    // printer_constraints: if Some, max 32768 chars
    if let Some(ref pc) = design.printer_constraints {
        check!(validation::require_max_len(pc, 32768, "printer_constraints"));
    }

    // generated_file_cid: if Some, max 256 chars
    if let Some(ref cid) = design.generated_file_cid {
        check!(validation::require_max_len(cid, 256, "generated_file_cid"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a SemanticMatchEntry
fn validate_semantic_match(m: SemanticMatchEntry) -> ExternResult<ValidateCallbackResult> {
    // similarity_score: in range 0.0..1.0 (also catches NaN/Inf)
    check!(validation::require_in_range(m.similarity_score, 0.0, 1.0, "similarity_score"));

    // matched_bindings: max 128 items, each max 256 chars
    check!(validation::require_max_vec_len(&m.matched_bindings, 128, "matched_bindings"));
    for binding in &m.matched_bindings {
        check!(validation::require_max_len(binding, 256, "matched binding"));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an OptimizationResultEntry
fn validate_optimization_result(o: OptimizationResultEntry) -> ExternResult<ValidateCallbackResult> {
    // energy_preference: max 256 chars
    check!(validation::require_max_len(&o.energy_preference, 256, "energy_preference"));

    // parameter_adjustments: max 32768 chars
    check!(validation::require_max_len(&o.parameter_adjustments, 32768, "parameter_adjustments"));

    // improvement_metrics: max 32768 chars
    check!(validation::require_max_len(&o.improvement_metrics, 32768, "improvement_metrics"));

    // local_materials: max 64 items
    check!(validation::require_max_vec_len(&o.local_materials, 64, "local_materials"));

    // local_printers: max 64 items
    check!(validation::require_max_vec_len(&o.local_printers, 64, "local_printers"));

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test helpers
    // =========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn valid_intent() -> HdcIntentEntry {
        HdcIntentEntry {
            description: "Design a cup holder".to_string(),
            vector_dimensions: 16384,
            vector_hash: "abc123hash".to_string(),
            semantic_bindings: vec![SerializedBinding {
                concept: "cup".to_string(),
                role: "Base".to_string(),
                weight: 0.8,
            }],
            generation_method: "auto".to_string(),
            language: "en".to_string(),
            author: fake_agent(),
            created_at: Timestamp::now(),
        }
    }

    fn valid_generated_design() -> GeneratedDesignEntry {
        GeneratedDesignEntry {
            intent_hash: fake_action_hash(),
            base_design_hash: None,
            parametric_config: r#"{"width": 50}"#.to_string(),
            material_constraints: vec!["PLA".to_string()],
            printer_constraints: None,
            generated_file_cid: None,
            confidence_score: 0.85,
            generation_time_ms: 1200,
            created_at: Timestamp::now(),
        }
    }

    fn valid_semantic_match() -> SemanticMatchEntry {
        SemanticMatchEntry {
            query_intent_hash: fake_action_hash(),
            matched_design_hash: fake_action_hash(),
            similarity_score: 0.92,
            matched_bindings: vec!["cup".to_string()],
            searched_at: Timestamp::now(),
        }
    }

    fn valid_optimization_result() -> OptimizationResultEntry {
        OptimizationResultEntry {
            original_design_hash: fake_action_hash(),
            optimized_for: OptimizationTarget::CostReduction,
            local_materials: vec![fake_action_hash()],
            local_printers: vec![fake_action_hash()],
            energy_preference: "solar".to_string(),
            parameter_adjustments: r#"{"infill": 20}"#.to_string(),
            improvement_metrics: r#"{"cost": -0.15}"#.to_string(),
            created_at: Timestamp::now(),
        }
    }

    fn is_valid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    // =========================================================================
    // HdcIntentEntry tests
    // =========================================================================

    #[test]
    fn valid_intent_passes() {
        assert!(is_valid(validate_hdc_intent(valid_intent())));
    }

    #[test]
    fn empty_description_rejected() {
        let mut intent = valid_intent();
        intent.description = "   ".to_string();
        assert!(is_invalid(validate_hdc_intent(intent)));
    }

    #[test]
    fn description_too_long_rejected() {
        let mut intent = valid_intent();
        intent.description = "x".repeat(4097);
        assert!(is_invalid(validate_hdc_intent(intent)));
    }

    #[test]
    fn vector_dimensions_invalid_rejected() {
        // Too small
        let mut intent = valid_intent();
        intent.vector_dimensions = 512;
        assert!(is_invalid(validate_hdc_intent(intent)));

        // Not power of 2
        let mut intent2 = valid_intent();
        intent2.vector_dimensions = 10000;
        assert!(is_invalid(validate_hdc_intent(intent2)));

        // 8192 should pass (power of 2 in range)
        let mut intent3 = valid_intent();
        intent3.vector_dimensions = 8192;
        assert!(is_valid(validate_hdc_intent(intent3)));

        // Too large
        let mut intent4 = valid_intent();
        intent4.vector_dimensions = 32768;
        assert!(is_invalid(validate_hdc_intent(intent4)));
    }

    #[test]
    fn too_many_semantic_bindings_rejected() {
        let mut intent = valid_intent();
        intent.semantic_bindings = (0..129)
            .map(|i| SerializedBinding {
                concept: format!("concept_{}", i),
                role: "Base".to_string(),
                weight: 0.5,
            })
            .collect();
        assert!(is_invalid(validate_hdc_intent(intent)));
    }

    #[test]
    fn binding_weight_nan_rejected() {
        let mut intent = valid_intent();
        intent.semantic_bindings = vec![SerializedBinding {
            concept: "test".to_string(),
            role: "Base".to_string(),
            weight: f32::NAN,
        }];
        assert!(is_invalid(validate_hdc_intent(intent)));
    }

    // =========================================================================
    // GeneratedDesignEntry tests
    // =========================================================================

    #[test]
    fn valid_generated_design_passes() {
        assert!(is_valid(validate_generated_design(valid_generated_design())));
    }

    #[test]
    fn nan_confidence_score_rejected() {
        let mut design = valid_generated_design();
        design.confidence_score = f32::NAN;
        assert!(is_invalid(validate_generated_design(design)));
    }

    #[test]
    fn parametric_config_too_long_rejected() {
        let mut design = valid_generated_design();
        design.parametric_config = "x".repeat(32769);
        assert!(is_invalid(validate_generated_design(design)));
    }

    // =========================================================================
    // SemanticMatchEntry tests
    // =========================================================================

    #[test]
    fn valid_semantic_match_passes() {
        assert!(is_valid(validate_semantic_match(valid_semantic_match())));
    }

    #[test]
    fn nan_similarity_score_rejected() {
        let mut m = valid_semantic_match();
        m.similarity_score = f32::NAN;
        assert!(is_invalid(validate_semantic_match(m)));
    }

    // =========================================================================
    // OptimizationResultEntry tests
    // =========================================================================

    #[test]
    fn valid_optimization_result_passes() {
        assert!(is_valid(validate_optimization_result(valid_optimization_result())));
    }

    // =========================================================================
    // Link tag validation tests
    // =========================================================================

    #[test]
    fn test_link_tag_at_max_passes() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(result.is_err()); // Err(()) means "no validation issue found"
    }

    #[test]
    fn test_link_tag_over_max_rejected() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(msg)) if msg.contains("link tag")));
    }
}
