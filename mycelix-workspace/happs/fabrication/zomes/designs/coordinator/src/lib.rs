// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Designs Coordinator Zome
//!
//! This zome provides the coordinator functions for managing designs
//! in the Mycelix Fabrication hApp. It includes CRUD operations,
//! versioning, forking, discovery, and parametric operations.

use hdk::prelude::*;
use designs_integrity::*;
use fabrication_common::*;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static CONFIG: RefCell<Option<FabricationConfig>> = const { RefCell::new(None) };
}

fn get_config() -> FabricationConfig {
    CONFIG.with(|c| {
        c.borrow_mut()
            .get_or_insert_with(|| {
                dna_info()
                    .map(|info| FabricationConfig::from_properties_or_default(info.modifiers.properties.bytes()))
                    .unwrap_or_default()
            })
            .clone()
    })
}

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
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetDesignsByAuthorInput {
    pub author: AgentPubKey,
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetDesignsByCategoryInput {
    pub category: DesignCategory,
    pub pagination: Option<PaginationInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetFeaturedDesignsInput {
    pub pagination: Option<PaginationInput>,
}

/// Alias for backward compatibility with existing callers
pub type HashListInput = HashPaginationInput;

// =============================================================================
// CRUD OPERATIONS
// =============================================================================

// =============================================================================
// RATE LIMITING
// =============================================================================

fn rate_limit_anchor(agent: &AgentPubKey) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("rate_limit:{}", agent).into_bytes(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn enforce_rate_limit(caller: &AgentPubKey) -> ExternResult<()> {
    let cfg = get_config();
    let max_ops = cfg.rate_limit_max_ops as usize;
    let window_micros = cfg.rate_limit_window_secs as i64 * 1_000_000;

    let anchor = rate_limit_anchor(caller)?;
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::RateLimitBucket)?,
        GetStrategy::default(),
    )?;

    let now = sys_time()?;
    let window_start = now.as_micros() - window_micros;

    let recent_count = links
        .iter()
        .filter(|l| l.timestamp.as_micros() >= window_start)
        .count();

    if recent_count >= max_ops {
        return Err(FabricationError::RateLimited {
            max_ops: cfg.rate_limit_max_ops,
            window_secs: cfg.rate_limit_window_secs,
        }.to_wasm_error());
    }

    create_link(anchor.clone(), anchor, LinkTypes::RateLimitBucket, ())?;
    Ok(())
}

fn rate_limit_caller() -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    enforce_rate_limit(&agent)
}

// =============================================================================
// CRUD OPERATIONS
// =============================================================================

/// Create a new design
#[hdk_extern]
pub fn create_design(input: CreateDesignInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Generate default HDC vector if not provided
    let dim = get_config().hdc_dimensions as usize;
    let intent_vector = input.intent_vector.unwrap_or_else(|| HdcHypervector {
        dimensions: dim as u32,
        vector: vec![0; dim],
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Design,
        event_type: FabricationEventType::DesignCreated,
        payload: format!(r#"{{"hash":"{}"}}"#, action_hash),
    });

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
    let original = get(input.original_action_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &input.original_action_hash))?;

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
        return Err(FabricationError::unauthorized("update_design", "Only the author can update a design"));
    }

    let now = sys_time()?;

    let updated_design = Design {
        id: original_design.id,
        title: input.title.unwrap_or(original_design.title),
        description: input.description.unwrap_or(original_design.description),
        category: input.category.unwrap_or(original_design.category),
        intent_vector: input.intent_vector.unwrap_or(original_design.intent_vector),
        parametric_schema: input.parametric_schema.or(original_design.parametric_schema),
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

    let new_hash = update_entry(input.original_action_hash, EntryTypes::Design(updated_design))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Design,
        event_type: FabricationEventType::DesignUpdated,
        payload: format!(r#"{{"hash":"{}"}}"#, new_hash),
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated design".to_string()
    )))
}

/// Delete a design (marks as deleted, doesn't remove from DHT)
#[hdk_extern]
pub fn delete_design(hash: ActionHash) -> ExternResult<ActionHash> {
    let design_record = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &hash))?;

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
        return Err(FabricationError::unauthorized("delete_design", "Only the author can delete a design"));
    }

    let delete_hash = delete_entry(hash.clone())?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Design,
        event_type: FabricationEventType::DesignDeleted,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    Ok(delete_hash)
}

// =============================================================================
// FILE MANAGEMENT
// =============================================================================

/// Add a file to a design (only the design author may add files)
#[hdk_extern]
pub fn add_design_file(input: AddFileInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let uploader = agent_info()?.agent_initial_pubkey;

    // Verify caller is the design author
    let design_record = get(input.design_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Design not found".to_string())))?;
    let design: Design = design_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not parse design".to_string())))?;
    if design.author != uploader {
        return Err(FabricationError::unauthorized("add_design_file", "Only the design author can add files"));
    }

    let now = sys_time()?;

    let file_entry = DesignFileEntry {
        design_hash: input.design_hash.clone(),
        file: input.file,
        uploader,
        uploaded_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let file_hash = create_entry(EntryTypes::DesignFile(file_entry))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Design,
        event_type: FabricationEventType::FileAdded,
        payload: format!(r#"{{"hash":"{}"}}"#, file_hash),
    });

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
pub fn get_design_files(input: HashListInput) -> ExternResult<PaginatedResponse<Record>> {
    let links = get_links(
        LinkQuery::try_new(input.hash, LinkTypes::DesignToFiles)?, GetStrategy::default(),
    )?;

    let mut files = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                files.push(record);
            }
        }
    }

    Ok(paginate(files, input.pagination.as_ref()))
}

// =============================================================================
// VERSIONING & FORKING
// =============================================================================

/// Fork a design to create a derivative
#[hdk_extern]
pub fn fork_design(input: ForkDesignInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let parent_record = get(input.parent_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Parent design not found".to_string()
        )))?;

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
        License::CreativeCommons(CCVariant::BYND | CCVariant::BYNCND) => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "This Creative Commons license does not allow derivatives".to_string()
            )));
        }
        License::CreativeCommons(_) => {}
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
        title: input.title.unwrap_or_else(|| format!("Fork of {}", parent.title)),
        description: input.description.unwrap_or_else(|| {
            format!("{}\n\n---\nForked from: {}", parent.description, parent.title)
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

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Design,
        event_type: FabricationEventType::DesignForked,
        payload: format!(r#"{{"hash":"{}"}}"#, child_hash),
    });

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
pub fn get_design_history(input: HashListInput) -> ExternResult<PaginatedResponse<Record>> {
    let details = get_details(input.hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &input.hash))?;

    match details {
        Details::Entry(entry_details) => {
            let mut records: Vec<Record> = Vec::new();
            for signed_action in entry_details.actions {
                let action_hash = signed_action.hashed.hash.clone();
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    records.push(record);
                }
            }
            records.sort_by(|a, b| {
                a.action()
                    .timestamp()
                    .cmp(&b.action().timestamp())
            });
            Ok(paginate(records, input.pagination.as_ref()))
        }
        _ => Ok(paginate(vec![], input.pagination.as_ref())),
    }
}

/// Get all forks of a design
#[hdk_extern]
pub fn get_design_forks(input: HashListInput) -> ExternResult<PaginatedResponse<Record>> {
    let links = get_links(
        LinkQuery::try_new(input.hash, LinkTypes::ParentToForks)?, GetStrategy::default(),
    )?;

    let mut forks = Vec::new();
    for link in links {
        if let Some(target_hash) = link.target.into_action_hash() {
            if let Some(record) = get(target_hash, GetOptions::default())? {
                forks.push(record);
            }
        }
    }

    Ok(paginate(forks, input.pagination.as_ref()))
}

// =============================================================================
// DISCOVERY
// =============================================================================

/// Get all designs by an author
#[hdk_extern]
pub fn get_designs_by_author(input: GetDesignsByAuthorInput) -> ExternResult<PaginatedResponse<Record>> {
    let links = get_links(
        LinkQuery::try_new(input.author, LinkTypes::AuthorToDesigns)?, GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(paginate(designs, input.pagination.as_ref()))
}

/// Get all designs in a category
#[hdk_extern]
pub fn get_designs_by_category(input: GetDesignsByCategoryInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = category_anchor(&input.category)?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::CategoryToDesigns)?, GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(paginate(designs, input.pagination.as_ref()))
}

/// Search designs with various filters
#[hdk_extern]
pub fn search_designs(query: DesignSearchQuery) -> ExternResult<PaginatedResponse<Record>> {
    // Get all designs
    let anchor = all_designs_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllDesigns)?, GetStrategy::default())?;

    let mut results = Vec::new();

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(design) = record
                    .entry()
                    .to_app_option::<Design>()
                    .ok()
                    .flatten()
                {
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

    Ok(paginate(results, query.pagination.as_ref()))
}

/// Get featured designs
#[hdk_extern]
pub fn get_featured_designs(input: GetFeaturedDesignsInput) -> ExternResult<PaginatedResponse<Record>> {
    let anchor = featured_designs_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::FeaturedDesigns)?, GetStrategy::default(),
    )?;

    let mut designs = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                designs.push(record);
            }
        }
    }

    Ok(paginate(designs, input.pagination.as_ref()))
}

// =============================================================================
// PARAMETRIC OPERATIONS
// =============================================================================

/// Get parametric configuration for a design
#[hdk_extern]
pub fn get_parameters(hash: ActionHash) -> ExternResult<Option<ParametricSchema>> {
    let record = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &hash))?;

    let design: Design = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse design".to_string()
        )))?;

    Ok(design.parametric_schema)
}

/// Generate a parametric variant by delegating to the symthaea CSG zome.
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
    /// The action hash of the GeneratedDesignEntry created in the symthaea
    /// zome, if the cross-zome call succeeded.
    pub csg_record_hash: Option<ActionHash>,
}

/// Mirror of symthaea coordinator's `GenerateVariantInput` for the cross-zome
/// call.  Field names must match exactly so that MessagePack serialization on
/// the caller side deserializes correctly on the callee side.
#[derive(Serialize, Deserialize, Debug)]
struct SymthaeaVariantInput {
    pub base_design_hash: ActionHash,
    pub intent_modifiers: Vec<SerializedBindingForCall>,
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>,
}

/// Minimal mirror of `SerializedBinding` from the symthaea integrity crate.
/// We only need the wire-format fields; the type lives in the symthaea crate
/// so we cannot import it directly from the designs coordinator.
#[derive(Serialize, Deserialize, Debug)]
struct SerializedBindingForCall {
    pub concept: String,
    pub role: String,
    pub weight: f32,
}

/// Convert the caller's flat `HashMap<String, ParameterValue>` into the
/// symthaea zome's intent-modifier + material-constraint representation.
///
/// Rules:
/// - Every parameter becomes a `SerializedBinding` with the parameter key as
///   `concept`.
/// - The `role` is derived from the value type:
///   - `Number` / `Integer` → "dimension"
///   - `Boolean`            → "feature"
///   - `String` → "material" when the key contains "material",
///     "modifier" otherwise
/// - Parameters whose key contains "material" or whose `ParameterValue` is a
///   `String` with key containing "material" are also collected into
///   `material_constraints` as plain strings.
/// - `weight` is 1.0 for all generated bindings (the symthaea zome uses it
///   only for tie-breaking in HDC cosine similarity, so a uniform weight is
///   correct when the caller does not specify priority).
fn params_to_symthaea_input(
    design_hash: ActionHash,
    parameters: &HashMap<String, ParameterValue>,
) -> SymthaeaVariantInput {
    let mut intent_modifiers: Vec<SerializedBindingForCall> = Vec::new();
    let mut material_constraints: Vec<String> = Vec::new();

    for (key, value) in parameters {
        let key_lower = key.to_lowercase();
        let is_material_key = key_lower.contains("material") || key_lower.contains("filament");

        let (role, concept_value) = match value {
            ParameterValue::Number(n) => {
                if !n.is_finite() {
                    continue; // Skip non-finite parameter values
                }
                ("dimension".to_string(), format!("{}={}", key, n))
            }
            ParameterValue::Integer(i) => ("dimension".to_string(), format!("{}={}", key, i)),
            ParameterValue::Boolean(b) => ("feature".to_string(), format!("{}={}", key, b)),
            ParameterValue::String(s) => {
                if is_material_key {
                    material_constraints.push(s.clone());
                    ("material".to_string(), s.clone())
                } else {
                    ("modifier".to_string(), format!("{}={}", key, s))
                }
            }
        };

        intent_modifiers.push(SerializedBindingForCall {
            concept: concept_value,
            role,
            weight: 1.0,
        });
    }

    SymthaeaVariantInput {
        base_design_hash: design_hash,
        intent_modifiers,
        material_constraints,
        printer_constraints: None,
    }
}

#[hdk_extern]
pub fn generate_variant(input: GenerateVariantInput) -> ExternResult<GeneratedVariant> {
    // -------------------------------------------------------------------------
    // 1. Validate: design must exist on DHT and have a parametric schema.
    // -------------------------------------------------------------------------
    let record = get(input.design_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Design", &input.design_hash))?;

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

    // -------------------------------------------------------------------------
    // 2. Attempt cross-zome call to symthaea's parametric variant generator.
    // -------------------------------------------------------------------------
    let symthaea_input =
        params_to_symthaea_input(input.design_hash.clone(), &input.parameters);

    let csg_result = call(
        CallTargetCell::Local,
        ZomeName::from("symthaea"),
        FunctionName::from("generate_parametric_variant"),
        None,
        &symthaea_input,
    );

    // -------------------------------------------------------------------------
    // 3. Interpret the cross-zome result.
    // -------------------------------------------------------------------------
    match csg_result {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            // Decode the returned Record to extract the action hash.  The
            // symthaea zome returns ExternResult<Record>, so the outer
            // ExternResult is already unwrapped by the HDK call machinery;
            // the payload is the serialized Record.
            let csg_record: Record = extern_io
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode symthaea response: {}",
                    e
                ))))?;

            let csg_hash = csg_record.action_address().clone();

            Ok(GeneratedVariant {
                design_hash: input.design_hash,
                parameters_used: input.parameters,
                // generated_file_cid is populated later by the native
                // fabrication-kernel service; the Record on DHT carries it.
                output_file: None,
                generation_status: "csg_resolved".to_string(),
                csg_record_hash: Some(csg_hash),
            })
        }

        // Any non-Ok ZomeCallResponse or an HDK-level error falls through to
        // the fallback branch.  The `generation_status` field communicates
        // the failure reason to callers; verbose logging is not needed here.
        Ok(ZomeCallResponse::NetworkError(_))
        | Ok(ZomeCallResponse::Unauthorized(..))
        | Ok(_) => Ok(GeneratedVariant {
            design_hash: input.design_hash,
            parameters_used: input.parameters,
            output_file: None,
            generation_status: "csg_unavailable".to_string(),
            csg_record_hash: None,
        }),
        Err(_) => {
            // HDK-level error (e.g., symthaea zome absent from DNA) — still a
            // graceful fallback so callers continue to function without CSG.
            Ok(GeneratedVariant {
                design_hash: input.design_hash,
                parameters_used: input.parameters,
                output_file: None,
                generation_status: "csg_unavailable".to_string(),
                csg_record_hash: None,
            })
        }
    }
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
    format!("design_{}_{}", now.as_micros(), &agent[..8.min(agent.len())])
}

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", name).into_bytes()
    ));
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

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_create_design_input_serde() {
        let input = CreateDesignInput {
            title: "Test Bracket".to_string(),
            description: "A simple test bracket".to_string(),
            category: DesignCategory::Parts,
            intent_vector: Some(HdcHypervector {
                dimensions: 8192,
                vector: vec![0i8; 8192],
                semantic_bindings: vec![],
                generation_method: HdcMethod::ManualEncoding,
            }),
            parametric_schema: None,
            constraint_graph: None,
            material_compatibility: vec![],
            circularity_score: 0.5,
            embodied_energy_kwh: 1.0,
            repair_manifest: None,
            license: License::PublicDomain,
            safety_class: SafetyClass::Class0Decorative,
        };

        let json = serde_json::to_string(&input).expect("serialization failed");
        let decoded: CreateDesignInput =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(decoded.title, "Test Bracket");
        assert_eq!(decoded.description, "A simple test bracket");
        assert!(matches!(decoded.category, DesignCategory::Parts));
        assert_eq!(decoded.circularity_score, 0.5);
        assert_eq!(decoded.embodied_energy_kwh, 1.0);
        assert!(matches!(decoded.license, License::PublicDomain));
        assert!(matches!(decoded.safety_class, SafetyClass::Class0Decorative));
        assert!(decoded.parametric_schema.is_none());
        assert!(decoded.constraint_graph.is_none());
        assert!(decoded.repair_manifest.is_none());
        assert!(decoded.material_compatibility.is_empty());

        let hv = decoded.intent_vector.expect("intent_vector missing");
        assert_eq!(hv.dimensions, 8192);
        assert_eq!(hv.vector.len(), 8192);
        assert!(matches!(hv.generation_method, HdcMethod::ManualEncoding));
    }

    #[test]
    fn test_update_design_input_serde() {
        let original_action_hash = ActionHash::from_raw_36(vec![0u8; 36]);

        let input = UpdateDesignInput {
            original_action_hash: original_action_hash.clone(),
            title: Some("Updated Title".to_string()),
            description: Some("Updated description".to_string()),
            category: Some(DesignCategory::Tools),
            intent_vector: None,
            parametric_schema: None,
            constraint_graph: None,
            material_compatibility: None,
            circularity_score: Some(0.75),
            embodied_energy_kwh: None,
            repair_manifest: None,
            license: Some(License::OpenHardware),
            safety_class: Some(SafetyClass::Class1Functional),
            epistemic: Some(DesignEpistemic {
                manufacturability: 0.8,
                safety: 0.9,
                usability: 0.7,
            }),
        };

        let json = serde_json::to_string(&input).expect("serialization failed");
        let decoded: UpdateDesignInput =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(decoded.original_action_hash, original_action_hash);
        assert_eq!(decoded.title.as_deref(), Some("Updated Title"));
        assert_eq!(decoded.description.as_deref(), Some("Updated description"));
        assert!(matches!(decoded.category, Some(DesignCategory::Tools)));
        assert_eq!(decoded.circularity_score, Some(0.75));
        assert!(matches!(decoded.license, Some(License::OpenHardware)));
        assert!(matches!(decoded.safety_class, Some(SafetyClass::Class1Functional)));
        assert!(decoded.intent_vector.is_none());
        assert!(decoded.parametric_schema.is_none());
        assert!(decoded.repair_manifest.is_none());

        let ep = decoded.epistemic.expect("epistemic missing");
        assert_eq!(ep.manufacturability, 0.8);
        assert_eq!(ep.safety, 0.9);
        assert_eq!(ep.usability, 0.7);
    }

    #[test]
    fn test_design_search_query_serde() {
        let input = DesignSearchQuery {
            query: Some("bracket".to_string()),
            category: Some(DesignCategory::Repair),
            safety_class: Some(SafetyClass::Class2LoadBearing),
            min_circularity: Some(0.6),
            license: Some(License::CreativeCommons(CCVariant::BY)),
            limit: Some(20),
            pagination: Some(PaginationInput {
                offset: 0,
                limit: 20,
            }),
        };

        let json = serde_json::to_string(&input).expect("serialization failed");
        let decoded: DesignSearchQuery =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(decoded.query.as_deref(), Some("bracket"));
        assert!(matches!(decoded.category, Some(DesignCategory::Repair)));
        assert!(matches!(decoded.safety_class, Some(SafetyClass::Class2LoadBearing)));
        assert_eq!(decoded.min_circularity, Some(0.6));
        assert_eq!(decoded.limit, Some(20));

        let lic = decoded.license.expect("license missing");
        assert!(matches!(lic, License::CreativeCommons(CCVariant::BY)));

        let page = decoded.pagination.expect("pagination missing");
        assert_eq!(page.offset, 0);
        assert_eq!(page.limit, 20);
    }

    #[test]
    fn test_fork_design_input_serde() {
        let parent_hash = ActionHash::from_raw_36(vec![0u8; 36]);

        let input = ForkDesignInput {
            parent_hash: parent_hash.clone(),
            modification_notes: "Increased wall thickness for outdoor use".to_string(),
            title: Some("Outdoor Bracket (Fork)".to_string()),
            description: Some("Weather-resistant variant".to_string()),
            intent_modifications: Some(vec![SemanticBinding {
                concept: "weatherproof".to_string(),
                role: BindingRole::Modifier,
                weight: 0.9,
            }]),
        };

        let json = serde_json::to_string(&input).expect("serialization failed");
        let decoded: ForkDesignInput =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(decoded.parent_hash, parent_hash);
        assert_eq!(
            decoded.modification_notes,
            "Increased wall thickness for outdoor use"
        );
        assert_eq!(decoded.title.as_deref(), Some("Outdoor Bracket (Fork)"));
        assert_eq!(
            decoded.description.as_deref(),
            Some("Weather-resistant variant")
        );

        let mods = decoded.intent_modifications.expect("intent_modifications missing");
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].concept, "weatherproof");
        assert!(matches!(mods[0].role, BindingRole::Modifier));
        assert_eq!(mods[0].weight, 0.9);
    }

    #[test]
    fn test_generate_variant_input_serde() {
        let design_hash = ActionHash::from_raw_36(vec![0u8; 36]);

        let mut parameters: HashMap<String, ParameterValue> = HashMap::new();
        parameters.insert("wall_thickness".to_string(), ParameterValue::Number(3.0));
        parameters.insert("height".to_string(), ParameterValue::Integer(50));
        parameters.insert("include_holes".to_string(), ParameterValue::Boolean(true));
        parameters.insert(
            "material_grade".to_string(),
            ParameterValue::String("PETG-CF".to_string()),
        );

        let input = GenerateVariantInput {
            design_hash: design_hash.clone(),
            parameters,
        };

        let json = serde_json::to_string(&input).expect("serialization failed");
        let decoded: GenerateVariantInput =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(decoded.design_hash, design_hash);
        assert_eq!(decoded.parameters.len(), 4);

        assert!(matches!(
            decoded.parameters.get("wall_thickness"),
            Some(ParameterValue::Number(v)) if (*v - 3.0).abs() < f64::EPSILON
        ));
        assert!(matches!(
            decoded.parameters.get("height"),
            Some(ParameterValue::Integer(50))
        ));
        assert!(matches!(
            decoded.parameters.get("include_holes"),
            Some(ParameterValue::Boolean(true))
        ));
        assert!(matches!(
            decoded.parameters.get("material_grade"),
            Some(ParameterValue::String(s)) if s == "PETG-CF"
        ));
    }

    // =========================================================================
    // params_to_symthaea_input tests
    // =========================================================================

    fn make_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    #[test]
    fn test_params_number_produces_dimension_role() {
        let mut params = HashMap::new();
        params.insert("width".to_string(), ParameterValue::Number(50.0));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.intent_modifiers.len(), 1);
        assert_eq!(result.intent_modifiers[0].role, "dimension");
        assert_eq!(result.intent_modifiers[0].concept, "width=50");
        assert_eq!(result.intent_modifiers[0].weight, 1.0);
    }

    #[test]
    fn test_params_boolean_produces_feature_role() {
        let mut params = HashMap::new();
        params.insert("chamfered".to_string(), ParameterValue::Boolean(true));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.intent_modifiers.len(), 1);
        assert_eq!(result.intent_modifiers[0].role, "feature");
        assert!(result.intent_modifiers[0].concept.contains("chamfered=true"));
    }

    #[test]
    fn test_params_material_key_collected() {
        let mut params = HashMap::new();
        params.insert("material_type".to_string(), ParameterValue::String("PLA".to_string()));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.material_constraints, vec!["PLA"]);
        assert_eq!(result.intent_modifiers[0].role, "material");
    }

    #[test]
    fn test_params_string_non_material_produces_modifier_role() {
        let mut params = HashMap::new();
        params.insert("color".to_string(), ParameterValue::String("red".to_string()));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.intent_modifiers[0].role, "modifier");
        assert!(result.material_constraints.is_empty());
    }

    #[test]
    fn test_params_nan_skipped() {
        let mut params = HashMap::new();
        params.insert("width".to_string(), ParameterValue::Number(f64::NAN));
        params.insert("height".to_string(), ParameterValue::Number(25.0));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.intent_modifiers.len(), 1);
        assert_eq!(result.intent_modifiers[0].concept, "height=25");
    }

    #[test]
    fn test_params_infinity_skipped() {
        let mut params = HashMap::new();
        params.insert("depth".to_string(), ParameterValue::Number(f64::INFINITY));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert!(result.intent_modifiers.is_empty());
    }

    #[test]
    fn test_params_integer_produces_dimension_role() {
        let mut params = HashMap::new();
        params.insert("count".to_string(), ParameterValue::Integer(4));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.intent_modifiers[0].role, "dimension");
        assert_eq!(result.intent_modifiers[0].concept, "count=4");
    }

    #[test]
    fn test_params_filament_key_is_material() {
        let mut params = HashMap::new();
        params.insert("filament_type".to_string(), ParameterValue::String("PETG".to_string()));
        let result = params_to_symthaea_input(make_hash(), &params);
        assert_eq!(result.material_constraints, vec!["PETG"]);
        assert_eq!(result.intent_modifiers[0].role, "material");
    }

    #[test]
    fn test_params_empty_map() {
        let params = HashMap::new();
        let result = params_to_symthaea_input(make_hash(), &params);
        assert!(result.intent_modifiers.is_empty());
        assert!(result.material_constraints.is_empty());
    }
}
