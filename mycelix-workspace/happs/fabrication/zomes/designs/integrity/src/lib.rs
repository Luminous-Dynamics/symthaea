//! Designs Integrity Zome
//!
//! This zome defines the entry types and validation rules for designs
//! in the Mycelix Fabrication hApp. It implements HDC-encoded parametric
//! designs for generative manufacturing.

use hdi::prelude::*;
use fabrication_common::*;
use fabrication_common::validation;

/// Entry types for the designs zome
#[allow(clippy::large_enum_variant)] // Holochain entry serialization requires unboxed variants
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A 3D printable design with HDC parametric intelligence
    #[entry_type(visibility = "public")]
    Design(Design),
    /// A design file (STL, STEP, etc.)
    #[entry_type(visibility = "public")]
    DesignFile(DesignFileEntry),
    /// Design modification/fork
    #[entry_type(visibility = "public")]
    DesignModification(DesignModification),
}

/// Link types for the designs zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from author agent to their designs
    AuthorToDesigns,
    /// Link from category anchor to designs
    CategoryToDesigns,
    /// Link from parent design to forks
    ParentToForks,
    /// Link from design to its files
    DesignToFiles,
    /// Link from design to verifications
    DesignToVerifications,
    /// Link for all designs discovery
    AllDesigns,
    /// Link for featured designs
    FeaturedDesigns,
    /// Per-agent rate limiting bucket
    RateLimitBucket,
}

/// A 3D printable design with HDC parametric intelligence
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Design {
    /// Unique identifier
    pub id: String,
    /// Design title
    pub title: String,
    /// Description of the design
    pub description: String,
    /// Category for organization
    pub category: DesignCategory,

    // === HDC PARAMETRIC INTELLIGENCE ===
    /// 10,000-dim semantic encoding of design intent
    pub intent_vector: HdcHypervector,
    /// Parametric schema for generative manufacturing
    pub parametric_schema: Option<ParametricSchema>,
    /// Dimensional constraint relationships
    pub constraint_graph: Option<ConstraintGraph>,
    /// HDC material compatibility vectors
    pub material_compatibility: Vec<MaterialBinding>,

    // === STATIC FILES (Legacy Support) ===
    /// List of design files (linked separately)
    pub file_count: u32,

    // === METABOLIC INTEGRATION ===
    /// Recyclability score (0.0-1.0)
    pub circularity_score: f32,
    /// Manufacturing energy cost in kWh
    pub embodied_energy_kwh: f32,
    /// Links to parent products for repair
    pub repair_manifest: Option<RepairManifest>,

    /// License terms
    pub license: License,
    /// Safety classification
    pub safety_class: SafetyClass,
    /// Epistemic dimensions
    pub epistemic: DesignEpistemic,
    /// Creator's public key
    pub author: AgentPubKey,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last update timestamp
    pub updated_at: Timestamp,
}

/// A design file entry stored in DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DesignFileEntry {
    /// Design this file belongs to
    pub design_hash: ActionHash,
    /// File metadata
    pub file: DesignFile,
    /// Uploader
    pub uploader: AgentPubKey,
    /// Upload timestamp
    pub uploaded_at: Timestamp,
}

/// A modification/fork of a design
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DesignModification {
    /// Original design being forked
    pub parent_hash: ActionHash,
    /// New forked design
    pub child_hash: ActionHash,
    /// Description of modifications
    pub modification_notes: String,
    /// Who made the modification
    pub modifier: AgentPubKey,
    /// When the modification was made
    pub modified_at: Timestamp,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
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
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            ..
        } => validate_create_link(link_type, base_address, target_address, tag),
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

/// Validate entry creation
fn validate_create_entry(entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::Design(design) => validate_design(design),
        EntryTypes::DesignFile(file) => validate_design_file(file),
        EntryTypes::DesignModification(modification) => validate_modification(modification),
    }
}

/// Validate a design entry
fn validate_design(design: Design) -> ExternResult<ValidateCallbackResult> {
    // --- String field validation ---
    check!(validation::require_non_empty(&design.id, "Design ID"));
    check!(validation::require_max_len(&design.id, 64, "Design ID"));

    check!(validation::require_non_empty(&design.title, "Design title"));
    check!(validation::require_max_len(&design.title, 256, "Design title"));

    check!(validation::require_max_len(&design.description, 4096, "Design description"));

    // --- HDC hypervector validation ---
    // Must be a power of 2 in [4096, 16384], matching symthaea integrity
    if !design.intent_vector.dimensions.is_power_of_two()
        || design.intent_vector.dimensions < 4096
        || design.intent_vector.dimensions > 16384
    {
        return Ok(ValidateCallbackResult::Invalid(
            "HDC hypervector dimensions must be a power of 2 between 4096 and 16384".to_string(),
        ));
    }

    if design.intent_vector.vector.len() != design.intent_vector.dimensions as usize {
        return Ok(ValidateCallbackResult::Invalid(
            "HDC vector length must match dimensions".to_string(),
        ));
    }

    // --- Semantic bindings validation ---
    check!(validation::require_max_vec_len(
        &design.intent_vector.semantic_bindings, 128, "semantic_bindings"
    ));
    for binding in &design.intent_vector.semantic_bindings {
        check!(validation::require_max_len(&binding.concept, 256, "binding concept"));
        check!(validation::require_in_range(binding.weight, 0.0, 1.0, "binding weight"));
    }

    // --- Float field validation ---
    check!(validation::require_in_range(design.circularity_score, 0.0, 1.0, "circularity_score"));

    check!(validation::require_finite(design.embodied_energy_kwh, "embodied_energy_kwh"));
    if design.embodied_energy_kwh < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "embodied_energy_kwh must be >= 0".to_string(),
        ));
    }

    // --- Collection size limits ---
    check!(validation::require_max_vec_len(
        &design.material_compatibility, 64, "material_compatibility"
    ));

    // Sanity check on file_count
    if design.file_count > 10_000 {
        return Ok(ValidateCallbackResult::Invalid(
            "file_count exceeds reasonable limit of 10000".to_string(),
        ));
    }

    // --- Epistemic scores ---
    check!(validation::require_in_range(
        design.epistemic.manufacturability, 0.0, 1.0, "epistemic.manufacturability"
    ));
    check!(validation::require_in_range(
        design.epistemic.safety, 0.0, 1.0, "epistemic.safety"
    ));
    check!(validation::require_in_range(
        design.epistemic.usability, 0.0, 1.0, "epistemic.usability"
    ));

    // --- Material compatibility item validation ---
    for mat in &design.material_compatibility {
        check!(validation::require_in_range(
            mat.compatibility, 0.0, 1.0, "material compatibility"
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a design file entry
fn validate_design_file(file: DesignFileEntry) -> ExternResult<ValidateCallbackResult> {
    // Filename: non-empty, max 256
    check!(validation::require_non_empty(&file.file.filename, "filename"));
    check!(validation::require_max_len(&file.file.filename, 256, "filename"));

    // IPFS CID: non-empty, max 256
    check!(validation::require_non_empty(&file.file.ipfs_cid, "IPFS CID"));
    check!(validation::require_max_len(&file.file.ipfs_cid, 256, "IPFS CID"));

    // Checksum: non-empty, max 128
    check!(validation::require_non_empty(&file.file.checksum_sha256, "checksum"));
    check!(validation::require_max_len(&file.file.checksum_sha256, 128, "checksum"));

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a design modification
fn validate_modification(modification: DesignModification) -> ExternResult<ValidateCallbackResult> {
    // Modification notes: non-empty, max 4096
    check!(validation::require_non_empty(&modification.modification_notes, "Modification notes"));
    check!(validation::require_max_len(&modification.modification_notes, 4096, "Modification notes"));

    // Parent and child must be different
    if modification.parent_hash == modification.child_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Parent and child design must be different".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
    // Validate tag length for all link types (default: 256 bytes)
    let max_len = match link_type {
        LinkTypes::AuthorToDesigns => 256,
        LinkTypes::CategoryToDesigns => 256,
        LinkTypes::ParentToForks => 256,
        LinkTypes::DesignToFiles => 256,
        LinkTypes::DesignToVerifications => 256,
        LinkTypes::AllDesigns => 256,
        LinkTypes::FeaturedDesigns => 256,
        LinkTypes::RateLimitBucket => 256,
    };
    check!(validation::require_max_tag_len(&tag, max_len, &format!("{:?}", link_type)));

    match link_type {
        LinkTypes::AuthorToDesigns => {
            // Any agent can link their own designs
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::CategoryToDesigns => {
            // Category links are always valid
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::ParentToForks => {
            // Fork links should be created with modification entry
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::DesignToFiles => {
            // File links are always valid
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::DesignToVerifications => {
            // Verification links are always valid
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AllDesigns => {
            // All designs discovery link
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::FeaturedDesigns => {
            // Featured designs link
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::RateLimitBucket => Ok(ValidateCallbackResult::Valid),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a valid Design with 8192-dim vector for tests.
    fn valid_design() -> Design {
        Design {
            id: "test-design-001".to_string(),
            title: "Test Bracket".to_string(),
            description: "A simple test bracket for unit tests".to_string(),
            category: DesignCategory::Parts,
            intent_vector: HdcHypervector {
                dimensions: 8192,
                vector: vec![1i8; 8192],
                semantic_bindings: vec![],
                generation_method: HdcMethod::ManualEncoding,
            },
            parametric_schema: None,
            constraint_graph: None,
            material_compatibility: vec![],
            file_count: 1,
            circularity_score: 0.8,
            embodied_energy_kwh: 2.5,
            repair_manifest: None,
            license: License::PublicDomain,
            safety_class: SafetyClass::Class0Decorative,
            epistemic: DesignEpistemic {
                manufacturability: 0.9,
                safety: 0.95,
                usability: 0.85,
            },
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            created_at: Timestamp::from_micros(0),
            updated_at: Timestamp::from_micros(0),
        }
    }

    /// Helper: build a valid DesignFileEntry for tests.
    fn valid_design_file() -> DesignFileEntry {
        DesignFileEntry {
            design_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            file: DesignFile {
                filename: "bracket.stl".to_string(),
                format: FileFormat::STL,
                ipfs_cid: "QmTest1234567890".to_string(),
                size_bytes: 1024,
                checksum_sha256: "abc123def456".to_string(),
            },
            uploader: AgentPubKey::from_raw_36(vec![0u8; 36]),
            uploaded_at: Timestamp::from_micros(0),
        }
    }

    /// Helper: build a valid DesignModification for tests.
    fn valid_modification() -> DesignModification {
        DesignModification {
            parent_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            child_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            modification_notes: "Increased wall thickness for durability".to_string(),
            modifier: AgentPubKey::from_raw_36(vec![0u8; 36]),
            modified_at: Timestamp::from_micros(0),
        }
    }

    // ---- Design validation tests ----

    #[test]
    fn test_valid_design_passes() {
        let result = validate_design(valid_design()).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_empty_id_rejected() {
        let mut d = valid_design();
        d.id = "".to_string();
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Design ID")));
    }

    #[test]
    fn test_id_too_long_rejected() {
        let mut d = valid_design();
        d.id = "x".repeat(65);
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Design ID")));
    }

    #[test]
    fn test_title_too_long_rejected() {
        let mut d = valid_design();
        d.title = "x".repeat(257);
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Design title")));
    }

    #[test]
    fn test_nan_circularity_rejected() {
        let mut d = valid_design();
        d.circularity_score = f32::NAN;
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("circularity_score")));
    }

    #[test]
    fn test_nan_epistemic_rejected() {
        let mut d = valid_design();
        d.epistemic.manufacturability = f32::NAN;
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("epistemic.manufacturability")));
    }

    #[test]
    fn test_negative_energy_rejected() {
        let mut d = valid_design();
        d.embodied_energy_kwh = -1.0;
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("embodied_energy_kwh")));
    }

    #[test]
    fn test_too_many_materials_rejected() {
        let mut d = valid_design();
        d.material_compatibility = (0..65)
            .map(|_| MaterialBinding {
                material: MaterialType::PLA,
                compatibility: 0.9,
                requirements: vec![],
            })
            .collect();
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("material_compatibility")));
    }

    #[test]
    fn test_vector_dimension_mismatch_rejected() {
        let mut d = valid_design();
        // Set dimensions to 8192 but only provide 100 values
        d.intent_vector.dimensions = 8192;
        d.intent_vector.vector = vec![1i8; 100];
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("vector length must match")));
    }

    #[test]
    fn test_binding_weight_nan_rejected() {
        let mut d = valid_design();
        d.intent_vector.semantic_bindings = vec![SemanticBinding {
            concept: "bracket".to_string(),
            role: BindingRole::Base,
            weight: f32::NAN,
        }];
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("binding weight")));
    }

    // ---- DesignFile validation tests ----

    #[test]
    fn test_empty_filename_rejected() {
        let mut f = valid_design_file();
        f.file.filename = "".to_string();
        let result = validate_design_file(f).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("filename")));
    }

    // ---- DesignModification validation tests ----

    #[test]
    fn test_empty_modification_notes_rejected() {
        let mut m = valid_modification();
        m.modification_notes = "".to_string();
        let result = validate_modification(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Modification notes")));
    }

    #[test]
    fn test_parent_equals_child_rejected() {
        let mut m = valid_modification();
        let same_hash = ActionHash::from_raw_36(vec![42u8; 36]);
        m.parent_hash = same_hash.clone();
        m.child_hash = same_hash;
        let result = validate_modification(m).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("Parent and child")));
    }

    // ---- HDC dimension validation tests ----

    #[test]
    fn test_hdc_dim_16384_passes() {
        let mut d = valid_design();
        d.intent_vector.dimensions = 16384;
        d.intent_vector.vector = vec![1i8; 16384];
        let result = validate_design(d).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_hdc_dim_4096_passes() {
        let mut d = valid_design();
        d.intent_vector.dimensions = 4096;
        d.intent_vector.vector = vec![1i8; 4096];
        let result = validate_design(d).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_hdc_dim_non_power_of_2_rejected() {
        let mut d = valid_design();
        d.intent_vector.dimensions = 10000;
        d.intent_vector.vector = vec![1i8; 10000];
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("power of 2")));
    }

    #[test]
    fn test_hdc_dim_too_small_rejected() {
        let mut d = valid_design();
        d.intent_vector.dimensions = 2048;
        d.intent_vector.vector = vec![1i8; 2048];
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("4096 and 16384")));
    }

    #[test]
    fn test_hdc_dim_too_large_rejected() {
        let mut d = valid_design();
        d.intent_vector.dimensions = 32768;
        d.intent_vector.vector = vec![1i8; 32768];
        let result = validate_design(d).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(msg) if msg.contains("4096 and 16384")));
    }

    // ---- Link tag validation tests ----

    #[test]
    fn test_link_tag_at_max_length_passes() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(result.is_err()); // Err(()) means "no validation issue found"
    }

    #[test]
    fn test_link_tag_over_max_length_rejected() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validation::require_max_tag_len(&tag, 256, "test");
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(msg)) if msg.contains("link tag")));
    }
}
