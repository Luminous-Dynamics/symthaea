// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Designs Coordinator Zome
//!
//! This zome provides the coordinator functions for managing designs
//! in the Mycelix Fabrication hApp. It includes CRUD operations,
//! versioning, forking, discovery, and parametric operations.

use designs_integrity::*;
use fabrication_common::*;
use hdk::prelude::*;
use std::collections::HashMap;

/// Input for creating a new design
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateDesignInput {
    pub title: String,
    pub description: String,
    pub category: DesignCategory,
    pub intent_vector: Option<HdcHypervector>,
    pub parametric_schema: Option<ParametricSchema>,
    pub constraint_graph: Option<ConstraintGraph>,
    pub material_compatibility: Vec<MaterialBinding>,
    pub circularity_score: f32,
    pub embodied_energy_kwh: f32,
    pub repair_manifest: Option<RepairManifest>,
    pub license: License,
    pub safety_class: SafetyClass,
}

/// Input for updating a design
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDesignInput {
    pub original_action_hash: ActionHash,
    pub title: Option<String>,
    pub description: Option<String>,
    pub category: Option<DesignCategory>,
    pub intent_vector: Option<HdcHypervector>,
    pub parametric_schema: Option<ParametricSchema>,
    pub constraint_graph: Option<ConstraintGraph>,
    pub material_compatibility: Option<Vec<MaterialBinding>>,
    pub circularity_score: Option<f32>,
    pub embodied_energy_kwh: Option<f32>,
    pub repair_manifest: Option<RepairManifest>,
    pub license: Option<License>,
    pub safety_class: Option<SafetyClass>,
    pub epistemic: Option<DesignEpistemic>,
}

/// Input for adding a file to a design
#[derive(Serialize, Deserialize, Debug)]
pub struct AddFileInput {
    pub design_hash: ActionHash,
    pub file: DesignFile,
}

/// Input for forking a design
#[derive(Serialize, Deserialize, Debug)]
pub struct ForkDesignInput {
    pub parent_hash: ActionHash,
    pub modification_notes: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub intent_modifications: Option<Vec<SemanticBinding>>,
}

/// Search query for designs
#[derive(Serialize, Deserialize, Debug)]
pub struct DesignSearchQuery {
    pub query: Option<String>,
    pub category: Option<DesignCategory>,
    pub safety_class: Option<SafetyClass>,
    pub min_circularity: Option<f32>,
    pub license: Option<License>,
    pub limit: Option<u32>,
}

// =============================================================================
// CRUD OPERATIONS
// =============================================================================

/// Create a new design
#[hdk_extern]
pub fn create_design(input: CreateDesignInput) -> ExternResult<Record> {
    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Generate default HDC vector if not provided
    let intent_vector = input.intent_vector.unwrap_or_else(|| HdcHypervector {
        dimensions: 10000,
        vector: vec![0; 10000],
        semantic_bindings: vec![],
        generation_method: HdcMethod::ManualEncoding,
    });

    let design = Design {
        id: generate_id(),
        title: input.title,
        description: input.description,
        category: input.category.clone(),
        intent_vector,
        parametric_schema: input.parametric_schema,
        constraint_graph: input.constraint_graph,
        material_compatibility: input.material_compatibility,
        file_count: 0,
        circularity_score: input.circularity_score,
        embodied_energy_kwh: input.embodied_energy_kwh,
        repair_manifest: input.repair_manifest,
        license: input.license,
        safety_class: input.safety_class,
        epistemic: DesignEpistemic::default(),
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let action_hash = create_entry(EntryTypes::Design(design.clone()))?;

    // Create links for discovery
    create_link(
        author.clone(),
        action_hash.clone(),
        LinkTypes::AuthorToDesigns,
        (),
    )?;

    // Link to category anchor
    let cat_anchor = category_anchor(&input.category)?;
    create_link(
        cat_anchor,
        action_hash.clone(),
        LinkTypes::CategoryToDesigns,
        (),
    )?;

    // Link to all designs anchor
    let all_anchor = all_designs_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllDesigns, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created design".to_string()
    )))
}

/// Get a design by its action hash
#[hdk_extern]
pub fn get_design(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Update an existing design
#[hdk_extern]
pub fn update_design(input: UpdateDesignInput) -> ExternResult<Record> {
    let original = get(input.original_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Design not found".to_string())),
    )?;

    let original_design: Design = original
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse design".to_string()
        )))?;

    // Verify author
    let author = agent_info()?.agent_initial_pubkey;
    if original_design.author != author {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the author can update a design".to_string()
        )));
    }

    let now = sys_time()?;

    let updated_design = Design {
        id: original_design.id,
        title: input.title.unwrap_or(original_design.title),
        description: input.description.unwrap_or(original_design.description),
        category: input.category.unwrap_or(original_design.category),
        intent_vector: input.intent_vector.unwrap_or(original_design.intent_vector),
        parametric_schema: input
            .parametric_schema
            .or(original_design.parametric_schema),
        constraint_graph: input.constraint_graph.or(original_design.constraint_graph),
        material_compatibility: input
            .material_compatibility
            .unwrap_or(original_design.material_compatibility),
        file_count: original_design.file_count,
        circularity_score: input
            .circularity_score
            .unwrap_or(original_design.circularity_score),
        embodied_energy_kwh: input
            .embodied_energy_kwh
            .unwrap_or(original_design.embodied_energy_kwh),
        repair_manifest: input.repair_manifest.or(original_design.repair_manifest),
        license: input.license.unwrap_or(original_design.license),
        safety_class: input.safety_class.unwrap_or(original_design.safety_class),
        epistemic: input.epistemic.unwrap_or(original_design.epistemic),
        author: original_design.author,
        created_at: original_design.created_at,
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let new_hash = update_entry(
        input.original_action_hash,
        EntryTypes::Design(updated_design),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated design".to_string()
    )))
}

/// Delete a design (marks as deleted, doesn't remove from DHT)
#[hdk_extern]
pub fn delete_design(hash: ActionHash) -> ExternResult<ActionHash> {
    let design_record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Design not found".to_string())
    ))?;

    let design: Design = design_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse design".to_string()
        )))?;

    // Verify author
    let author = agent_info()?.agent_initial_pubkey;
    if design.author != author {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the author can delete a design".to_string()
        )));
    }

    delete_entry(hash)
}

// =============================================================================
// FILE MANAGEMENT
// =============================================================================

/// Add a file to a design
#[hdk_extern]
pub fn add_design_file(input: AddFileInput) -> ExternResult<Record> {
    let uploader = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let file_entry = DesignFileEntry {
        design_hash: input.design_hash.clone(),
        file: input.file,
        uploader,
        uploaded_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let file_hash = create_entry(EntryTypes::DesignFile(file_entry))?;

    // Link file to design
    create_link(
        input.design_hash,
        file_hash.clone(),
        LinkTypes::DesignToFiles,
        (),
    )?;

    get(file_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created file".to_string()
    )))
}

/// Get all files for a design
#[hdk_extern]
pub fn get_design_files(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(design_hash, LinkTypes::DesignToFiles)?,
        GetStrategy::default(),
    )?;

    let mut files = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                files.push(record);
            }
        }
    }

    Ok(files)
}

// =============================================================================
// VERSIONING & FORKING
// =============================================================================

/// Fork a design to create a derivative
#[hdk_extern]
pub fn fork_design(input: ForkDesignInput) -> ExternResult<Record> {
    let parent_record = get(input.parent_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Parent design not found".to_string())),
    )?;

    let parent: Design = parent_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse parent design".to_string()
        )))?;

    // Check license allows forking
    match &parent.license {
        License::Proprietary => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Proprietary designs cannot be forked".to_string()
            )));
        }
        License::CreativeCommons(variant) => match variant {
            CCVariant::BYND | CCVariant::BYNCND => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "This Creative Commons license does not allow derivatives".to_string()
                )));
            }
            _ => {}
        },
        _ => {}
    }

    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Merge intent modifications if provided
    let mut new_intent = parent.intent_vector.clone();
    if let Some(modifications) = input.intent_modifications {
        for binding in modifications {
            new_intent.semantic_bindings.push(binding);
        }
        new_intent.generation_method = HdcMethod::LateralBinding;
    }

    let forked_design = Design {
        id: generate_id(),
        title: input
            .title
            .unwrap_or_else(|| format!("Fork of {}", parent.title)),
        description: input.description.unwrap_or_else(|| {
            format!(
                "{}\n\n---\nForked from: {}",
                parent.description, parent.title
            )
        }),
        category: parent.category,
        intent_vector: new_intent,
        parametric_schema: parent.parametric_schema,
        constraint_graph: parent.constraint_graph,
        material_compatibility: parent.material_compatibility,
        file_count: 0, // Forked design starts without files
        circularity_score: parent.circularity_score,
        embodied_energy_kwh: parent.embodied_energy_kwh,
        repair_manifest: parent.repair_manifest,
        license: parent.license, // Inherit license
        safety_class: parent.safety_class,
        epistemic: DesignEpistemic::default(), // Reset epistemic scores
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        updated_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let child_hash = create_entry(EntryTypes::Design(forked_design.clone()))?;

    // Create modification entry
    let modification = DesignModification {
        parent_hash: input.parent_hash.clone(),
        child_hash: child_hash.clone(),
        modification_notes: input.modification_notes,
        modifier: author.clone(),
        modified_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    create_entry(EntryTypes::DesignModification(modification))?;

    // Link parent to fork
    create_link(
        input.parent_hash,
        child_hash.clone(),
        LinkTypes::ParentToForks,
        (),
    )?;

    // Standard links
    create_link(author, child_hash.clone(), LinkTypes::AuthorToDesigns, ())?;

    let cat_anchor = category_anchor(&forked_design.category)?;
    create_link(
        cat_anchor,
        child_hash.clone(),
        LinkTypes::CategoryToDesigns,
        (),
    )?;

    let all_anchor = all_designs_anchor()?;
    create_link(all_anchor, child_hash.clone(), LinkTypes::AllDesigns, ())?;

    get(child_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve forked design".to_string()
    )))
}

/// Get the history of a design (all updates)
#[hdk_extern]
pub fn get_design_history(hash: ActionHash) -> ExternResult<Vec<Record>> {
    let details = get_details(hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Design not found".to_string())
    ))?;

    match details {
        Details::Entry(entry_details) => {
            let mut records: Vec<Record> = Vec::new();
            for signed_action in entry_details.actions {
                // Get the action hash and fetch the full record
                let action_hash = signed_action.hashed.hash.clone();
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    records.push(record);
                }
            }
            // Sort by timestamp
            records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
            Ok(records)
        }
        _ => Ok(vec![]),
    }
}

/// Get all forks of a design
#[hdk_extern]
pub fn get_design_forks(hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hash, LinkTypes::ParentToForks)?,
        GetStrategy::default(),
    )?;

    let mut forks = Vec::new();
    for link in links {
        if let Some(target_hash) = link.target.into_action_hash() {
            if let Some(record) = get(target_hash, GetOptions::default())? {
                forks.push(record);
            }
        }
    }

    Ok(forks)
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Get all designs by an author
#[hdk_extern]
pub fn get_designs_by_author(author: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(author, LinkTypes::AuthorToDesigns)?,
        GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(designs)
}

/// Get all designs in a category
#[hdk_extern]
pub fn get_designs_by_category(category: DesignCategory) -> ExternResult<Vec<Record>> {
    let anchor = category_anchor(&category)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::CategoryToDesigns)?,
        GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(designs)
}

/// Search designs with various filters
#[hdk_extern]
pub fn search_designs(query: DesignSearchQuery) -> ExternResult<Vec<Record>> {
    // Get all designs
    let anchor = all_designs_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllDesigns)?,
        GetStrategy::default(),
    )?;

    let limit = query.limit.unwrap_or(100) as usize;
    let mut results = Vec::new();

    for link in links {
        if results.len() >= limit {
            break;
        }

        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(design) = record.entry().to_app_option::<Design>().ok().flatten() {
                    // Apply filters
                    let mut matches = true;

                    if let Some(ref cat) = query.category {
                        if design.category != *cat {
                            matches = false;
                        }
                    }

                    if let Some(ref sc) = query.safety_class {
                        if design.safety_class != *sc {
                            matches = false;
                        }
                    }

                    if let Some(min_circ) = query.min_circularity {
                        if design.circularity_score < min_circ {
                            matches = false;
                        }
                    }

                    if let Some(ref q) = query.query {
                        let query_lower = q.to_lowercase();
                        if !design.title.to_lowercase().contains(&query_lower)
                            && !design.description.to_lowercase().contains(&query_lower)
                        {
                            matches = false;
                        }
                    }

                    if matches {
                        results.push(record);
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Get featured designs
#[hdk_extern]
pub fn get_featured_designs(limit: u32) -> ExternResult<Vec<Record>> {
    let anchor = featured_designs_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::FeaturedDesigns)?,
        GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links.into_iter().take(limit as usize) {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(designs)
}

// =============================================================================
// PARAMETRIC OPERATIONS
// =============================================================================

/// Get parametric configuration for a design
#[hdk_extern]
pub fn get_parameters(hash: ActionHash) -> ExternResult<Option<ParametricSchema>> {
    let record = get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Design not found".to_string()
    )))?;

    let design: Design = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse design".to_string()
        )))?;

    Ok(design.parametric_schema)
}

/// Generate a parametric variant (placeholder for actual generation)
#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateVariantInput {
    pub design_hash: ActionHash,
    pub parameters: HashMap<String, ParameterValue>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GeneratedVariant {
    pub design_hash: ActionHash,
    pub parameters_used: HashMap<String, ParameterValue>,
    pub output_file: Option<DesignFile>,
    pub generation_status: String,
}

#[hdk_extern]
pub fn generate_variant(input: GenerateVariantInput) -> ExternResult<GeneratedVariant> {
    // Validate the design exists and has parametric schema
    let record = get(input.design_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Design not found".to_string())
    ))?;

    let design: Design = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse design".to_string()
        )))?;

    if design.parametric_schema.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Design does not have a parametric schema".to_string()
        )));
    }

    // In a real implementation, this would:
    // 1. Fetch the template from IPFS
    // 2. Apply parameters using OpenSCAD/CadQuery
    // 3. Generate the output file
    // 4. Upload to IPFS
    // 5. Return the generated file info

    Ok(GeneratedVariant {
        design_hash: input.design_hash,
        parameters_used: input.parameters,
        output_file: None, // Would be populated by actual generation
        generation_status: "Parametric generation requires external processing".to_string(),
    })
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a unique ID
fn generate_id() -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!(
        "design_{}_{}",
        now.as_micros(),
        &agent[..8.min(agent.len())]
    )
}

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes =
        SerializedBytes::from(UnsafeBytes::from(format!("anchor:{}", name).into_bytes()));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

/// Get the anchor for a category
fn category_anchor(category: &DesignCategory) -> ExternResult<EntryHash> {
    make_anchor(&format!("category_{:?}", category))
}

/// Get the anchor for all designs
fn all_designs_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_designs")
}

/// Get the anchor for featured designs
fn featured_designs_anchor() -> ExternResult<EntryHash> {
    make_anchor("featured_designs")
}
