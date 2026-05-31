// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Designs Integrity Zome
//!
//! This zome defines the entry types and validation rules for designs
//! in the Mycelix Fabrication hApp. It implements HDC-encoded parametric
//! designs for generative manufacturing.

use fabrication_common::*;
use hdi::prelude::*;

/// Entry types for the designs zome
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
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            // Links can be deleted by their creators
            let _ = link_type;
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
    // ID must not be empty and should be reasonably short
    if design.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Design ID cannot be empty".to_string(),
        ));
    }
    if design.id.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Design ID cannot exceed 200 characters".to_string(),
        ));
    }

    // Title must not be empty
    if design.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Design title cannot be empty".to_string(),
        ));
    }

    // Title length limit
    if design.title.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Design title cannot exceed 200 characters".to_string(),
        ));
    }

    // Description length limit
    if design.description.len() > 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Design description cannot exceed 10000 characters".to_string(),
        ));
    }

    // Validate HDC hypervector dimensions
    if design.intent_vector.dimensions != 10000 {
        return Ok(ValidateCallbackResult::Invalid(
            "HDC hypervector must have 10000 dimensions".to_string(),
        ));
    }

    // Validate vector length matches dimensions
    if design.intent_vector.vector.len() != design.intent_vector.dimensions as usize {
        return Ok(ValidateCallbackResult::Invalid(
            "HDC vector length must match dimensions".to_string(),
        ));
    }

    // Validate circularity score range
    if design.circularity_score < 0.0 || design.circularity_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circularity score must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Validate embodied energy is non-negative
    if design.embodied_energy_kwh < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Embodied energy cannot be negative".to_string(),
        ));
    }

    // Limit number of material compatibility entries to prevent abuse
    if design.material_compatibility.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Design has too many material compatibility entries (max 64)".to_string(),
        ));
    }

    // Sanity check on file_count
    if design.file_count > 10_000 {
        return Ok(ValidateCallbackResult::Invalid(
            "file_count exceeds reasonable limit of 10000".to_string(),
        ));
    }

    // Validate epistemic scores
    if design.epistemic.manufacturability < 0.0
        || design.epistemic.manufacturability > 1.0
        || design.epistemic.safety < 0.0
        || design.epistemic.safety > 1.0
        || design.epistemic.usability < 0.0
        || design.epistemic.usability > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Epistemic scores must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a design file entry
fn validate_design_file(file: DesignFileEntry) -> ExternResult<ValidateCallbackResult> {
    // Filename must not be empty
    if file.file.filename.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Filename cannot be empty".to_string(),
        ));
    }

    // IPFS CID must not be empty
    if file.file.ipfs_cid.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "IPFS CID cannot be empty".to_string(),
        ));
    }

    // Checksum must not be empty
    if file.file.checksum_sha256.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Checksum cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a design modification
fn validate_modification(modification: DesignModification) -> ExternResult<ValidateCallbackResult> {
    // Modification notes should not be empty
    if modification.modification_notes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Modification notes cannot be empty".to_string(),
        ));
    }

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
    _tag: LinkTag,
) -> ExternResult<ValidateCallbackResult> {
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
    }
}

/// Validate link deletion (reserved for future use)
#[allow(dead_code)]
fn validate_delete_link(
    _link_type: LinkTypes,
    _original_link: CreateLink,
) -> ExternResult<ValidateCallbackResult> {
    // Links can be deleted by their creators
    Ok(ValidateCallbackResult::Valid)
}
