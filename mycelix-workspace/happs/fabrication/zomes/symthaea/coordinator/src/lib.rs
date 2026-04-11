// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Coordinator Zome
//!
//! HDC (Hyperdimensional Computing) operations and AI-assisted design generation.
//! This zome handles:
//! - Natural language to HDC hypervector encoding (4096D continuous via FabHV)
//! - Lateral binding of semantic concepts (element-wise multiply in HDC space)
//! - Semantic similarity search (cosine similarity in HDC space)
//! - Parametric design generation
//! - Local condition optimization
//! - Repair prediction from sensor data

use hdk::prelude::*;
use symthaea_integrity::*;
use fabrication_common::hdc::{FabHV, FabTextEncoder, FAB_HDC_DIM};
use fabrication_common::*;

use std::cell::RefCell;

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
// HDC INTENT CREATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateIntentInput {
    pub description: String,
    pub language: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IntentResult {
    pub record: Record,
    pub bindings: Vec<SerializedBinding>,
    pub vector_hash: String,
}

/// Generate HDC hypervector from natural language description
/// e.g., "I need a bracket for a 12mm pipe that's weatherproof"
#[hdk_extern]
pub fn generate_intent_vector(input: CreateIntentInput) -> ExternResult<IntentResult> {
    rate_limit_caller()?;
    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Parse description into semantic bindings (supplementary metadata)
    let bindings = parse_semantic_bindings(&input.description);

    // Encode description into a real 4096D hypervector via FabTextEncoder
    let encoder = FabTextEncoder::new(FAB_HDC_DIM);
    let hv = encoder.encode(&input.description);

    // Generate deterministic hash from the actual HDC vector
    let vector_hash = generate_vector_hash(&hv);

    let intent = HdcIntentEntry {
        description: input.description,
        vector_dimensions: get_config().hdc_dimensions,
        vector_hash: vector_hash.clone(),
        semantic_bindings: bindings.clone(),
        generation_method: "symthaea_hdc".to_string(),
        language: input.language.unwrap_or_else(|| "en".to_string()),
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::HdcIntent(intent))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Symthaea,
        event_type: FabricationEventType::IntentGenerated,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(author, hash.clone(), LinkTypes::AuthorToIntents, ())?;

    // Link to global anchor for semantic_search
    let anchor = all_intents_anchor()?;
    create_link(anchor, hash.clone(), LinkTypes::AnchorToIntents, ())?;

    // Link to category anchor for partitioned search
    let category = extract_primary_category(&bindings);
    let cat_anchor = category_anchor(&category)?;
    create_link(cat_anchor, hash.clone(), LinkTypes::CategoryToIntents, ())?;

    let record = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Intent", &hash))?;

    Ok(IntentResult {
        record,
        bindings,
        vector_hash,
    })
}

/// Parse natural language into semantic bindings (supplementary metadata for
/// filtering and display; the real similarity comes from HDC vectors).
fn parse_semantic_bindings(description: &str) -> Vec<SerializedBinding> {
    let mut bindings = Vec::new();
    let lower = description.to_lowercase();

    // Object type detection (Base)
    let base_objects = [
        ("bracket", 1.0), ("mount", 1.0), ("holder", 1.0), ("clip", 1.0),
        ("adapter", 1.0), ("enclosure", 1.0), ("gear", 1.0), ("hinge", 1.0),
        ("knob", 1.0), ("handle", 1.0), ("hook", 1.0), ("stand", 1.0),
        ("cover", 1.0), ("case", 1.0), ("container", 1.0), ("box", 1.0),
    ];
    for (obj, weight) in base_objects {
        if lower.contains(obj) {
            bindings.push(SerializedBinding {
                concept: obj.to_string(),
                role: "Base".to_string(),
                weight,
            });
        }
    }

    // Dimensional constraints
    let dim_patterns = [
        ("mm", "Dimensional"), ("cm", "Dimensional"), ("inch", "Dimensional"),
        ("M3", "Dimensional"), ("M4", "Dimensional"), ("M5", "Dimensional"),
        ("M6", "Dimensional"), ("M8", "Dimensional"), ("M10", "Dimensional"),
    ];
    for (pattern, role) in dim_patterns {
        if lower.contains(&pattern.to_lowercase()) {
            // Extract number before unit
            let parts: Vec<&str> = lower.split_whitespace().collect();
            for part in parts {
                if part.contains(&pattern.to_lowercase()) {
                    bindings.push(SerializedBinding {
                        concept: part.to_string(),
                        role: role.to_string(),
                        weight: 0.9,
                    });
                }
            }
        }
    }

    // Material modifiers
    let materials = [
        ("pla", 0.8), ("petg", 0.8), ("abs", 0.8), ("tpu", 0.8),
        ("nylon", 0.8), ("food-safe", 0.9), ("food safe", 0.9),
    ];
    for (mat, weight) in materials {
        if lower.contains(mat) {
            bindings.push(SerializedBinding {
                concept: mat.to_string(),
                role: "Material".to_string(),
                weight,
            });
        }
    }

    // Property modifiers
    let modifiers = [
        ("weatherproof", 0.8), ("waterproof", 0.8), ("uv-resistant", 0.8),
        ("heat-resistant", 0.8), ("heavy-duty", 0.9), ("lightweight", 0.7),
        ("flexible", 0.8), ("rigid", 0.8), ("strong", 0.8),
    ];
    for (modifier, weight) in modifiers {
        if lower.contains(modifier) {
            bindings.push(SerializedBinding {
                concept: modifier.to_string(),
                role: "Modifier".to_string(),
                weight,
            });
        }
    }

    // Functional purpose
    let functions = [
        ("load-bearing", 0.9), ("decorative", 0.6), ("structural", 0.9),
        ("replacement", 0.8), ("repair", 0.8), ("custom", 0.7),
    ];
    for (func, weight) in functions {
        if lower.contains(func) {
            bindings.push(SerializedBinding {
                concept: func.to_string(),
                role: "Functional".to_string(),
                weight,
            });
        }
    }

    bindings
}

/// Generate a deterministic hash from the actual HDC hypervector using FNV-1a.
fn generate_vector_hash(hv: &FabHV) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for val in &hv.data {
        let bytes = val.to_le_bytes();
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
    }
    format!("hdc_{:016x}", hash)
}

// =============================================================================
// LATERAL BINDING (Vector Composition)
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct LateralBindInput {
    pub base_intent_hash: ActionHash,
    pub modifier_descriptions: Vec<String>,
}

/// Lateral binding: Combine base design with modifiers using HDC bind operation.
/// bracket_vector (x) 12mm_vector (x) weatherproof_vector
#[hdk_extern]
pub fn lateral_bind(input: LateralBindInput) -> ExternResult<IntentResult> {
    rate_limit_caller()?;
    // Get base intent
    let base_record = get(input.base_intent_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Intent", &input.base_intent_hash))?;

    let base_intent: HdcIntentEntry = base_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Parse error".into())))?;

    // Reconstruct the base HDC vector from the description (deterministic)
    let encoder = FabTextEncoder::new(FAB_HDC_DIM);
    let mut bound_hv = encoder.encode(&base_intent.description);

    // Combine descriptions (text-level for human readability)
    let combined_description = format!(
        "{} {}",
        base_intent.description,
        input.modifier_descriptions.join(" ")
    );

    // Bind each modifier's HDC vector with the base via element-wise multiply
    let mut all_bindings = base_intent.semantic_bindings.clone();
    for modifier in &input.modifier_descriptions {
        let modifier_hv = encoder.encode(modifier);
        bound_hv = bound_hv.bind(&modifier_hv);
        let modifier_bindings = parse_semantic_bindings(modifier);
        all_bindings.extend(modifier_bindings);
    }

    // Generate hash from the bound vector (not the text)
    let vector_hash = generate_vector_hash(&bound_hv);

    let author = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let combined_intent = HdcIntentEntry {
        description: combined_description,
        vector_dimensions: get_config().hdc_dimensions,
        vector_hash: vector_hash.clone(),
        semantic_bindings: all_bindings.clone(),
        generation_method: "lateral_binding".to_string(),
        language: base_intent.language,
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::HdcIntent(combined_intent))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Symthaea,
        event_type: FabricationEventType::IntentBound,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(author, hash.clone(), LinkTypes::AuthorToIntents, ())?;
    create_link(input.base_intent_hash, hash.clone(), LinkTypes::IntentToDesigns, ())?;

    let record = get(hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Intent", &hash))?;

    Ok(IntentResult {
        record,
        bindings: all_bindings,
        vector_hash,
    })
}

// =============================================================================
// SEMANTIC SEARCH
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct SemanticSearchInput {
    pub intent_hash: ActionHash,
    pub threshold: Option<f32>,
    pub limit: Option<u32>,
    /// If true, persist each match as a SemanticMatch DHT entry. Default: false.
    pub record_matches: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchResult {
    pub design_hash: ActionHash,
    pub similarity_score: f32,
    pub matched_bindings: Vec<String>,
}

/// Find designs by semantic similarity (cosine similarity in 4096D HDC space)
#[hdk_extern]
pub fn semantic_search(input: SemanticSearchInput) -> ExternResult<Vec<SearchResult>> {
    let threshold = input.threshold.unwrap_or(get_config().similarity_threshold);
    let limit = input.limit.unwrap_or(10) as usize;
    let should_record = input.record_matches.unwrap_or(false);

    // Get query intent
    let query_record = get(input.intent_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Intent", &input.intent_hash))?;

    let query_intent: HdcIntentEntry = query_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Parse error".into())))?;

    // Reconstruct query HDC vector from description (deterministic encoding)
    let encoder = FabTextEncoder::new(FAB_HDC_DIM);
    let query_hv = encoder.encode(&query_intent.description);

    // Get all intents and compute cosine similarity in HDC space
    let anchor = all_intents_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AnchorToIntents)?, GetStrategy::default())?;

    let mut results = Vec::new();
    let now = sys_time()?;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if hash == input.intent_hash {
                continue; // Skip self
            }
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(intent) = record.entry().to_app_option::<HdcIntentEntry>().ok().flatten() {
                    // Reconstruct candidate HDC vector and compute cosine similarity
                    let candidate_hv = encoder.encode(&intent.description);
                    let similarity = query_hv.similarity(&candidate_hv);

                    // Also compute binding overlap for metadata display
                    let matched = compute_binding_overlap(
                        &query_intent.semantic_bindings,
                        &intent.semantic_bindings,
                    );

                    if similarity >= threshold {
                        // Optionally persist the match as a DHT entry
                        if should_record {
                            let match_entry = SemanticMatchEntry {
                                query_intent_hash: input.intent_hash.clone(),
                                matched_design_hash: hash.clone(),
                                similarity_score: similarity,
                                matched_bindings: matched.clone(),
                                searched_at: Timestamp::from_micros(now.as_micros() as i64),
                            };
                            if let Err(e) = create_entry(EntryTypes::SemanticMatch(match_entry)) { debug!("Anchor creation warning: {:?}", e); }
                        }

                        results.push(SearchResult {
                            design_hash: hash,
                            similarity_score: similarity,
                            matched_bindings: matched,
                        });
                    }
                }
            }
        }
    }

    // Sort by similarity descending
    results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    results.truncate(limit);

    Ok(results)
}

/// Semantic search filtered by category — scans only intents in the same category.
#[hdk_extern]
pub fn semantic_search_by_category(input: SemanticSearchInput) -> ExternResult<Vec<SearchResult>> {
    let threshold = input.threshold.unwrap_or(get_config().similarity_threshold);
    let limit = input.limit.unwrap_or(10) as usize;
    let should_record = input.record_matches.unwrap_or(false);

    let query_record = get(input.intent_hash.clone(), GetOptions::default())?
        .ok_or(FabricationError::not_found("Intent", &input.intent_hash))?;
    let query_intent: HdcIntentEntry = query_record
        .entry().to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Parse error".into())))?;

    let encoder = FabTextEncoder::new(FAB_HDC_DIM);
    let query_hv = encoder.encode(&query_intent.description);

    // Get category anchor for the query's primary category
    let category = extract_primary_category(&query_intent.semantic_bindings);
    let anchor = category_anchor(&category)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::CategoryToIntents)?, GetStrategy::default())?;

    let mut results = Vec::new();
    let now = sys_time()?;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if hash == input.intent_hash { continue; }
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(intent) = record.entry().to_app_option::<HdcIntentEntry>().ok().flatten() {
                    let candidate_hv = encoder.encode(&intent.description);
                    let similarity = query_hv.similarity(&candidate_hv);
                    let matched = compute_binding_overlap(
                        &query_intent.semantic_bindings, &intent.semantic_bindings,
                    );
                    if similarity >= threshold {
                        if should_record {
                            let match_entry = SemanticMatchEntry {
                                query_intent_hash: input.intent_hash.clone(),
                                matched_design_hash: hash.clone(),
                                similarity_score: similarity,
                                matched_bindings: matched.clone(),
                                searched_at: Timestamp::from_micros(now.as_micros() as i64),
                            };
                            if let Err(e) = create_entry(EntryTypes::SemanticMatch(match_entry)) { debug!("Anchor creation warning: {:?}", e); }
                        }
                        results.push(SearchResult {
                            design_hash: hash,
                            similarity_score: similarity,
                            matched_bindings: matched,
                        });
                    }
                }
            }
        }
    }

    results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    results.truncate(limit);
    Ok(results)
}

/// LSH-accelerated semantic search — builds an in-memory LSH index from all
/// intents, then uses multi-probe approximate nearest neighbor to find candidates.
/// Falls back to brute-force when the index is small (<50 intents).
///
/// NOTE: The LSH index is rebuilt on each call because Holochain WASM zome calls
/// do not share state between invocations (thread_local is per-call).
/// For large datasets, consider using `semantic_search_by_category` instead,
/// which partitions the search space by category anchor.
#[derive(Serialize, Deserialize, Debug)]
pub struct LshSearchInput {
    pub query_description: String,
    pub threshold: Option<f32>,
    pub limit: Option<u32>,
}

#[hdk_extern]
pub fn semantic_search_lsh(input: LshSearchInput) -> ExternResult<Vec<SearchResult>> {
    use fabrication_common::lsh::{LshIndex, LSH_NUM_PLANES};

    let threshold = input.threshold.unwrap_or(get_config().similarity_threshold);
    let limit = input.limit.unwrap_or(10) as usize;

    let encoder = FabTextEncoder::new(FAB_HDC_DIM);
    let query_hv = encoder.encode(&input.query_description);

    // Build LSH index from all intents
    let anchor = all_intents_anchor()?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AnchorToIntents)?, GetStrategy::default())?;

    let mut index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 0xFAB0_0DC0);
    let mut intent_map: std::collections::HashMap<String, (ActionHash, HdcIntentEntry)> = std::collections::HashMap::new();

    for link in &links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash.clone(), GetOptions::default())? {
                if let Some(intent) = record.entry().to_app_option::<HdcIntentEntry>().ok().flatten() {
                    let hv = encoder.encode(&intent.description);
                    let id = hash.to_string();
                    index.insert(id.clone(), hv.data);
                    intent_map.insert(id, (hash, intent));
                }
            }
        }
    }

    // If small index, fall back to brute-force (LSH overhead not worth it)
    let candidates = if index.len() < 50 {
        // Brute-force: score all
        intent_map.iter().map(|(id, (_hash, intent))| {
            let hv = encoder.encode(&intent.description);
            let sim = query_hv.similarity(&hv);
            (id.clone(), sim)
        }).collect::<Vec<_>>()
    } else {
        // LSH multi-probe
        index.query_multiprobe(&query_hv, limit * 3) // over-fetch for threshold filtering
    };

    let mut results = Vec::new();
    for (id, similarity) in candidates {
        if similarity >= threshold {
            if let Some((hash, _intent)) = intent_map.get(&id) {
                results.push(SearchResult {
                    design_hash: hash.clone(),
                    similarity_score: similarity,
                    matched_bindings: vec![], // LSH search doesn't compute binding overlap
                });
            }
        }
    }

    results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    results.truncate(limit);
    Ok(results)
}

/// Compute binding overlap between two sets of semantic bindings.
/// Returns the list of concepts that appear in both sets with the same role.
/// This is supplementary metadata for display; the real similarity comes from
/// HDC cosine similarity.
fn compute_binding_overlap(
    a: &[SerializedBinding],
    b: &[SerializedBinding],
) -> Vec<String> {
    let mut matched = Vec::new();
    for binding_a in a {
        for binding_b in b {
            if binding_a.concept == binding_b.concept && binding_a.role == binding_b.role {
                matched.push(binding_a.concept.clone());
            }
        }
    }
    matched
}

// =============================================================================
// PARAMETRIC DESIGN GENERATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateVariantInput {
    pub base_design_hash: ActionHash,
    pub intent_modifiers: Vec<SerializedBinding>,
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>,
}

/// Generate parametric variant from intent + constraints
#[hdk_extern]
pub fn generate_parametric_variant(input: GenerateVariantInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let now = sys_time()?;

    // First create an intent from the modifiers
    let description = input.intent_modifiers
        .iter()
        .map(|b| format!("{} ({})", b.concept, b.role))
        .collect::<Vec<_>>()
        .join(", ");

    let intent_result = generate_intent_vector(CreateIntentInput {
        description,
        language: None,
    })?;

    // Build a CSG parametric config from the intent modifiers.
    // This describes the geometry tree that the fabrication kernel
    // resolves to mesh (STL/3MF) on the native-side service.
    let parametric_config = build_parametric_config(&input.intent_modifiers);

    // Confidence based on how many modifiers are structurally recognized
    let recognized = input.intent_modifiers.iter()
        .filter(|b| matches!(b.role.as_str(), "shape" | "dimension" | "material" | "feature" | "transform"))
        .count();
    let confidence = if input.intent_modifiers.is_empty() {
        0.5
    } else {
        0.5 + 0.5 * (recognized as f32 / input.intent_modifiers.len() as f32)
    };

    let generated = GeneratedDesignEntry {
        intent_hash: intent_result.record.action_address().clone(),
        base_design_hash: Some(input.base_design_hash.clone()),
        parametric_config,
        material_constraints: input.material_constraints,
        printer_constraints: input.printer_constraints,
        generated_file_cid: None, // Populated by native fabrication-kernel service
        confidence_score: confidence,
        generation_time_ms: 0,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::GeneratedDesign(generated))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Symthaea,
        event_type: FabricationEventType::VariantGenerated,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(
        input.base_design_hash,
        hash.clone(),
        LinkTypes::IntentToDesigns,
        (),
    )?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("GeneratedDesign", &hash))
}

// =============================================================================
// LOCAL OPTIMIZATION
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct OptimizeLocalInput {
    pub design_hash: ActionHash,
    pub local_materials: Vec<ActionHash>,
    pub local_printers: Vec<ActionHash>,
    pub energy_preference: String,
}

/// Optimize design for local conditions
#[hdk_extern]
pub fn optimize_for_local(input: OptimizeLocalInput) -> ExternResult<Record> {
    rate_limit_caller()?;
    let now = sys_time()?;

    // Calculate parameter adjustments based on local constraints
    let adjustments = calculate_local_adjustments(
        &input.local_materials,
        &input.local_printers,
        &input.energy_preference,
    );

    let improvement_metrics = calculate_improvement_metrics(&adjustments);

    let optimization = OptimizationResultEntry {
        original_design_hash: input.design_hash.clone(),
        optimized_for: OptimizationTarget::Combined,
        local_materials: input.local_materials,
        local_printers: input.local_printers,
        energy_preference: input.energy_preference,
        parameter_adjustments: serde_json::to_string(&adjustments).unwrap_or_else(|_| "{}".to_string()),
        improvement_metrics: serde_json::to_string(&improvement_metrics).unwrap_or_else(|_| "{}".to_string()),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::OptimizationResult(optimization))?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Symthaea,
        event_type: FabricationEventType::OptimizationCompleted,
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(
        input.design_hash,
        hash.clone(),
        LinkTypes::DesignToOptimizations,
        (),
    )?;

    get(hash.clone(), GetOptions::default())?.ok_or(FabricationError::not_found("Optimization", &hash))
}

fn calculate_local_adjustments(
    _materials: &[ActionHash],
    _printers: &[ActionHash],
    energy_pref: &str,
) -> std::collections::HashMap<String, String> {
    let mut adjustments = std::collections::HashMap::new();

    // Energy-based adjustments
    match energy_pref {
        "solar" | "renewable" => {
            adjustments.insert("print_speed".to_string(), "reduced_10%".to_string());
            adjustments.insert("infill_pattern".to_string(), "efficient".to_string());
        }
        "grid" => {
            adjustments.insert("print_time".to_string(), "off_peak".to_string());
        }
        _ => {}
    }

    adjustments.insert("local_optimized".to_string(), "true".to_string());
    adjustments
}

fn calculate_improvement_metrics(
    adjustments: &std::collections::HashMap<String, String>,
) -> std::collections::HashMap<String, f32> {
    let mut metrics = std::collections::HashMap::new();

    metrics.insert("energy_efficiency".to_string(), 0.15);
    metrics.insert("material_utilization".to_string(), 0.10);
    metrics.insert("local_economy_boost".to_string(), 0.20);

    if adjustments.contains_key("print_time") {
        metrics.insert("grid_friendliness".to_string(), 0.25);
    }

    metrics
}

// =============================================================================
// PARAMETRIC CONFIG BUILDER
// =============================================================================

/// Build a CSG parametric config JSON from intent modifiers.
/// This describes a geometry tree that the fabrication kernel resolves to mesh.
/// Format matches symthaea-fabrication-kernel CSGNode serialization.
fn build_parametric_config(modifiers: &[SerializedBinding]) -> String {
    let mut shape = "Cube";
    let mut scale = [1.0f32, 1.0, 1.0];
    let mut features: Vec<serde_json::Value> = Vec::new();

    for binding in modifiers {
        match binding.role.as_str() {
            "shape" => {
                shape = match binding.concept.to_lowercase().as_str() {
                    s if s.contains("cylinder") || s.contains("pipe") || s.contains("tube") => "Cylinder",
                    s if s.contains("sphere") || s.contains("ball") => "Sphere",
                    s if s.contains("cone") => "Cone",
                    s if s.contains("torus") || s.contains("ring") || s.contains("donut") => "Torus",
                    _ => "Cube",
                };
            }
            "dimension" => {
                // Parse dimension values like "12mm", "50x30x10"
                let nums: Vec<f32> = binding.concept
                    .split(|c: char| !c.is_ascii_digit() && c != '.')
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                match nums.len() {
                    1 => { scale = [nums[0] / 1000.0, nums[0] / 1000.0, nums[0] / 1000.0]; }
                    2 => { scale = [nums[0] / 1000.0, nums[1] / 1000.0, nums[0] / 1000.0]; }
                    3.. => { scale = [nums[0] / 1000.0, nums[1] / 1000.0, nums[2] / 1000.0]; }
                    _ => {}
                }
            }
            "feature" => {
                // Features like "mounting hole", "chamfer", "fillet"
                if binding.concept.to_lowercase().contains("hole") {
                    features.push(serde_json::json!({
                        "Boolean": {
                            "op": "Subtract",
                            "left": {"Primitive": "Cylinder"},
                            "right": {"Primitive": "Cylinder"}
                        }
                    }));
                }
            }
            _ => {}
        }
    }

    let mut tree = serde_json::json!({"Primitive": shape});

    // Apply scale transform if non-default
    if scale != [1.0, 1.0, 1.0] {
        tree = serde_json::json!({
            "Transform": {
                "node": tree,
                "transform": {
                    "scale": scale,
                    "rotate": [0.0, 0.0, 0.0],
                    "translate": [0.0, 0.0, 0.0]
                }
            }
        });
    }

    // Apply features via boolean operations
    for feature in features {
        tree = serde_json::json!({
            "Boolean": {
                "op": "Subtract",
                "left": tree,
                "right": feature
            }
        });
    }

    serde_json::json!({
        "csg_tree": tree,
        "version": "1.0",
        "kernel": "symthaea-fabrication-kernel"
    }).to_string()
}

// =============================================================================
// REPAIR PREDICTION
// =============================================================================

/// Per-asset-type MTBF table (hours). Looked up by asset_type field or falls
/// back to "general" (2000h). Values are industry baselines that callers can
/// override with measured data.
fn lookup_mtbf(asset_type: &str) -> u32 {
    match asset_type.to_lowercase().as_str() {
        "fdm_printer" | "3d_printer" => 3000,
        "sla_printer" | "resin_printer" => 2500,
        "cnc_mill" | "cnc_router" => 5000,
        "laser_cutter" => 4000,
        "power_drill" | "drill" => 1500,
        "impact_driver" => 1200,
        "circular_saw" | "saw" => 1800,
        "angle_grinder" | "grinder" => 1000,
        "compressor" => 6000,
        "motor" | "electric_motor" => 8000,
        "pump" => 4500,
        "hvac" | "heat_pump" => 7000,
        _ => 2000, // General fallback
    }
}

/// Per-component failure mode signatures.  Used to map dominant sensor
/// channel to the most likely failing component.
fn component_from_signature(signature: &SensorSignature) -> &'static str {
    // Prioritise by signal strength (strongest channel wins)
    let mut scores: Vec<(&str, f32)> = Vec::new();

    if signature.vibration_rms > 0.0 {
        scores.push(("bearing", signature.vibration_rms));
    }
    if signature.vibration_peak_freq > 0.0 {
        // High-frequency peaks → gear teeth or belt; low → imbalance
        if signature.vibration_peak_freq > 500.0 {
            scores.push(("gearbox", signature.vibration_peak_freq / 1000.0));
        } else {
            scores.push(("shaft_balance", signature.vibration_peak_freq / 500.0));
        }
    }
    if signature.temp_max > 60.0 {
        scores.push(("thermal_component", (signature.temp_max - 60.0) / 40.0));
    }
    if signature.torque_variance > 0.0 {
        scores.push(("chuck_mechanism", signature.torque_variance));
    }
    if signature.current_draw_delta > 0.0 {
        scores.push(("motor_winding", signature.current_draw_delta));
    }
    if signature.pressure_drop > 0.0 {
        scores.push(("seal", signature.pressure_drop));
    }
    if signature.stress_cycles > 0 {
        scores.push(("fatigue_joint", (signature.stress_cycles as f32) / 50_000.0));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.first().map(|(c, _)| *c).unwrap_or("unknown")
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PredictRepairInput {
    pub property_asset_hash: ActionHash,
    pub sensor_history: Vec<SensorReading>,
    pub usage_hours: u32,
    /// Asset type for MTBF lookup (e.g., "fdm_printer", "drill"). Falls back
    /// to "general" (2000h) when missing.
    #[serde(default)]
    pub asset_type: Option<String>,
    /// Override MTBF if the caller has measured data.
    #[serde(default)]
    pub mtbf_hours: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SensorReading {
    pub timestamp: i64,
    pub sensor_type: String,
    pub value: f32,
    pub unit: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RepairPredictionResult {
    pub predicted_component: String,
    pub failure_probability: f32,
    pub estimated_remaining_hours: u32,
    pub recommended_action: String,
    pub sensor_summary: SensorSummary,
    pub matching_repair_designs: Vec<ActionHash>,
}

/// Structured summary of the numeric sensor analysis
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SensorSummary {
    pub vibration_rms: f32,
    pub vibration_trend_slope: f32,
    pub vibration_peak_freq: f32,
    pub temp_mean: f32,
    pub temp_max: f32,
    pub temp_trend_slope: f32,
    pub torque_variance: f32,
    pub current_draw_delta: f32,
    pub pressure_drop: f32,
    pub stress_cycles: u32,
    pub degradation_score: f32,
}

/// Predict repair needs from digital twin data
#[hdk_extern]
pub fn predict_repair_needs(input: PredictRepairInput) -> ExternResult<RepairPredictionResult> {
    let analysis = analyze_sensor_degradation(&input.sensor_history);

    // MTBF: caller override > asset-type lookup > general fallback
    let base_mtbf = input.mtbf_hours.unwrap_or_else(|| {
        lookup_mtbf(input.asset_type.as_deref().unwrap_or("general"))
    });

    // Weibull-inspired failure probability:
    // P(t) = 1 - exp(-(t/η)^β) where η = MTBF, β = shape (sensor-adjusted)
    let t = input.usage_hours as f32;
    let eta = base_mtbf as f32;
    // β > 1 = wear-out, β < 1 = infant mortality; sensor degradation shifts β up
    let beta = 1.5 + analysis.degradation_score * 2.0; // Range [1.5, 3.5]
    let failure_probability = if eta > 0.0 {
        (1.0 - (-(t / eta).powf(beta)).exp()).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Remaining useful life: solve P(t+Δ)=0.9 for Δ
    let target_p = 0.9f32;
    let remaining_hours = if failure_probability < target_p && eta > 0.0 {
        // t_target = η × (-ln(1 - target_p))^(1/β)
        let t_target = eta * (-((1.0 - target_p).ln())).powf(1.0 / beta);
        if t_target > t { (t_target - t) as u32 } else { 0 }
    } else {
        0
    };

    let recommended_action = if failure_probability > 0.8 {
        "PrintReplacement"
    } else if failure_probability > 0.6 {
        "ScheduleReplacement"
    } else if failure_probability > 0.4 {
        "OrderMaterials"
    } else {
        "Monitor"
    }.to_string();

    let matching_designs = Vec::new();

    Ok(RepairPredictionResult {
        predicted_component: analysis.likely_component,
        failure_probability,
        estimated_remaining_hours: remaining_hours,
        recommended_action,
        sensor_summary: analysis.summary,
        matching_repair_designs: matching_designs,
    })
}

/// Internal: aggregated sensor signal envelope
struct SensorSignature {
    vibration_rms: f32,
    #[allow(dead_code)] // Reserved for trend analysis
    vibration_trend_slope: f32,
    vibration_peak_freq: f32,
    #[allow(dead_code)] // Reserved for thermal baseline
    temp_mean: f32,
    temp_max: f32,
    #[allow(dead_code)] // Reserved for trend analysis
    temp_trend_slope: f32,
    torque_variance: f32,
    current_draw_delta: f32,
    pressure_drop: f32,
    stress_cycles: u32,
}

struct DegradationAnalysis {
    degradation_score: f32,
    likely_component: String,
    summary: SensorSummary,
}

/// Real numeric sensor analysis: compute RMS, trend slopes (linear regression),
/// peak frequencies, and per-channel statistics from raw sensor readings.
fn analyze_sensor_degradation(readings: &[SensorReading]) -> DegradationAnalysis {
    if readings.is_empty() {
        return DegradationAnalysis {
            degradation_score: 0.5,
            likely_component: "unknown".to_string(),
            summary: SensorSummary {
                vibration_rms: 0.0, vibration_trend_slope: 0.0,
                vibration_peak_freq: 0.0, temp_mean: 0.0, temp_max: 0.0,
                temp_trend_slope: 0.0, torque_variance: 0.0,
                current_draw_delta: 0.0, pressure_drop: 0.0,
                stress_cycles: 0, degradation_score: 0.5,
            },
        };
    }

    // Partition readings by sensor type
    let mut vibration: Vec<(f32, f32)> = Vec::new(); // (time, value)
    let mut temperature: Vec<(f32, f32)> = Vec::new();
    let mut torque: Vec<f32> = Vec::new();
    let mut current: Vec<f32> = Vec::new();
    let mut pressure: Vec<f32> = Vec::new();
    let mut stress_cycles: u32 = 0;

    let t0 = readings.iter().map(|r| r.timestamp).min().unwrap_or(0) as f32;
    for r in readings {
        let t = (r.timestamp as f32 - t0) / 3600.0; // Relative hours
        match r.sensor_type.as_str() {
            "vibration" => vibration.push((t, r.value)),
            "temperature" => temperature.push((t, r.value)),
            "torque" => torque.push(r.value),
            "current" | "current_draw" => current.push(r.value),
            "pressure" => pressure.push(r.value),
            "stress_cycle" | "stress_cycles" => stress_cycles += r.value as u32,
            _ => {}
        }
    }

    // Vibration: RMS + trend slope + dominant frequency estimate
    let vibration_rms = rms(&vibration.iter().map(|(_, v)| *v).collect::<Vec<_>>());
    let vibration_trend_slope = linear_slope(&vibration);
    // Peak frequency: use zero-crossing rate as lightweight proxy
    let vibration_peak_freq = zero_crossing_rate(
        &vibration.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
        vibration.last().map(|(t, _)| *t).unwrap_or(1.0).max(0.001),
    );

    // Temperature: mean, max, trend slope
    let temp_values: Vec<f32> = temperature.iter().map(|(_, v)| *v).collect();
    let temp_mean = mean(&temp_values);
    let temp_max = temp_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let temp_max = if temp_max.is_finite() { temp_max } else { 0.0 };
    let temp_trend_slope = linear_slope(&temperature);

    // Torque: variance
    let torque_variance = variance(&torque);

    // Current: delta between first and last quartile means
    let current_draw_delta = if current.len() >= 4 {
        let q = current.len() / 4;
        let first_q: f32 = current[..q].iter().sum::<f32>() / q as f32;
        let last_q: f32 = current[current.len() - q..].iter().sum::<f32>() / q as f32;
        (last_q - first_q).max(0.0)
    } else {
        0.0
    };

    // Pressure: max drop from first reading
    let pressure_drop = if pressure.len() >= 2 {
        let first = pressure[0];
        pressure.iter().copied().map(|p| (first - p).max(0.0)).fold(0.0f32, f32::max)
    } else {
        0.0
    };

    let sig = SensorSignature {
        vibration_rms, vibration_trend_slope, vibration_peak_freq,
        temp_mean, temp_max, temp_trend_slope,
        torque_variance, current_draw_delta, pressure_drop, stress_cycles,
    };

    // Composite degradation score (0.0 = healthy, 1.0 = critical)
    // Each channel contributes proportionally to its severity
    let mut score = 0.0f32;
    let mut weight_sum = 0.0f32;

    // Vibration RMS > 5 mm/s is generally concerning for rotating machinery
    if vibration_rms > 0.0 {
        score += (vibration_rms / 10.0).clamp(0.0, 1.0) * 0.25;
        weight_sum += 0.25;
    }
    // Positive vibration trend (getting worse)
    if vibration_trend_slope > 0.0 {
        score += (vibration_trend_slope / 5.0).clamp(0.0, 1.0) * 0.15;
        weight_sum += 0.15;
    }
    // Temperature above 60°C is concerning for most components
    if temp_max > 0.0 {
        score += ((temp_max - 40.0) / 60.0).clamp(0.0, 1.0) * 0.15;
        weight_sum += 0.15;
    }
    // Rising temperature trend
    if temp_trend_slope > 0.0 {
        score += (temp_trend_slope / 10.0).clamp(0.0, 1.0) * 0.1;
        weight_sum += 0.1;
    }
    // Torque variance > 0.1 indicates mechanical looseness
    if torque_variance > 0.0 {
        score += (torque_variance / 0.5).clamp(0.0, 1.0) * 0.15;
        weight_sum += 0.15;
    }
    // Current draw increase indicates motor degradation
    if current_draw_delta > 0.0 {
        score += (current_draw_delta / 2.0).clamp(0.0, 1.0) * 0.1;
        weight_sum += 0.1;
    }
    // Pressure drop indicates seal failure
    if pressure_drop > 0.0 {
        score += (pressure_drop / 5.0).clamp(0.0, 1.0) * 0.1;
        weight_sum += 0.1;
    }

    let degradation_score = if weight_sum > 0.0 {
        (score / weight_sum).clamp(0.0, 1.0)
    } else {
        0.5 // No sensor data → uncertain
    };

    let likely_component = component_from_signature(&sig).to_string();

    let summary = SensorSummary {
        vibration_rms, vibration_trend_slope, vibration_peak_freq,
        temp_mean, temp_max, temp_trend_slope, torque_variance,
        current_draw_delta, pressure_drop, stress_cycles, degradation_score,
    };

    DegradationAnalysis { degradation_score, likely_component, summary }
}

// ── Numeric helpers ──────────────────────────────────────────────────────

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    let sum_sq: f32 = values.iter().map(|v| v * v).sum();
    (sum_sq / values.len() as f32).sqrt()
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() { return 0.0; }
    values.iter().sum::<f32>() / values.len() as f32
}

fn variance(values: &[f32]) -> f32 {
    if values.len() < 2 { return 0.0; }
    let m = mean(values);
    values.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / (values.len() - 1) as f32
}

/// Simple linear regression slope for (x, y) pairs
fn linear_slope(pairs: &[(f32, f32)]) -> f32 {
    if pairs.len() < 2 { return 0.0; }
    let n = pairs.len() as f32;
    let sum_x: f32 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f32 = pairs.iter().map(|(_, y)| y).sum();
    let sum_xy: f32 = pairs.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f32 = pairs.iter().map(|(x, _)| x * x).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 { return 0.0; }
    (n * sum_xy - sum_x * sum_y) / denom
}

/// Zero-crossing rate as a lightweight dominant frequency proxy.
/// Returns estimated frequency in Hz.
fn zero_crossing_rate(values: &[f32], duration_hours: f32) -> f32 {
    if values.len() < 3 || duration_hours <= 0.0 { return 0.0; }
    let m = mean(values);
    let crossings = values.windows(2)
        .filter(|w| (w[0] - m).signum() != (w[1] - m).signum())
        .count();
    // Each zero-crossing pair ≈ half a cycle
    let cycles = crossings as f32 / 2.0;
    cycles / (duration_hours * 3600.0) // Convert to Hz
}

// =============================================================================
// QUERIES
// =============================================================================

/// Input for paginated agent-scoped queries
#[derive(Serialize, Deserialize, Debug)]
pub struct MyIntentsInput {
    pub pagination: Option<PaginationInput>,
}

#[hdk_extern]
pub fn get_my_intents(input: MyIntentsInput) -> ExternResult<PaginatedResponse<Record>> {
    let author = agent_info()?.agent_initial_pubkey;
    let links = get_links(LinkQuery::try_new(author, LinkTypes::AuthorToIntents)?, GetStrategy::default())?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                results.push(record);
            }
        }
    }
    Ok(paginate(results, input.pagination.as_ref()))
}

#[hdk_extern]
pub fn get_design_optimizations(input: HashPaginationInput) -> ExternResult<PaginatedResponse<Record>> {
    let links = get_links(
        LinkQuery::try_new(input.hash, LinkTypes::DesignToOptimizations)?, GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                results.push(record);
            }
        }
    }
    Ok(paginate(results, input.pagination.as_ref()))
}

// =============================================================================
// HELPERS
// =============================================================================

/// Simple anchor helper - creates deterministic hash from string
fn make_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        format!("anchor:{}", name).into_bytes()
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}

fn all_intents_anchor() -> ExternResult<EntryHash> {
    make_anchor("all_intents")
}

/// Category anchor for partitioned HDC search.
/// Intents are linked to both `all_intents` and their primary category,
/// so `semantic_search_by_category` can scan a smaller subset.
fn category_anchor(category: &str) -> ExternResult<EntryHash> {
    make_anchor(&format!("category:{}", category.to_lowercase()))
}

/// Extract the primary category from semantic bindings.
/// Falls back to "general" if no recognized category is found.
fn extract_primary_category(bindings: &[SerializedBinding]) -> String {
    for binding in bindings {
        if binding.role == "Base" {
            let concept = binding.concept.to_lowercase();
            // Map common concepts to categories
            if ["bracket", "mount", "clip", "holder", "stand"].iter().any(|k| concept.contains(k)) {
                return "fasteners".to_string();
            }
            if ["gear", "pulley", "bearing", "shaft", "wheel"].iter().any(|k| concept.contains(k)) {
                return "mechanical".to_string();
            }
            if ["pipe", "tube", "fitting", "valve", "nozzle"].iter().any(|k| concept.contains(k)) {
                return "plumbing".to_string();
            }
            if ["case", "enclosure", "box", "housing", "cover"].iter().any(|k| concept.contains(k)) {
                return "enclosures".to_string();
            }
            if ["tool", "jig", "fixture", "gauge", "template"].iter().any(|k| concept.contains(k)) {
                return "tools".to_string();
            }
        }
    }
    "general".to_string()
}

// =============================================================================
// CONSCIOUSNESS-GATED FABRICATION
// =============================================================================

/// Consciousness tiers for fabrication gating. Higher consciousness enables
/// more creative/exploratory parametric generation; lower consciousness uses
/// proven templates with tighter constraints.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum CivicTier {
    /// Tier 0: Minimal awareness — strict templates only
    Reflexive,
    /// Tier 1: Pattern recognition — template + minor variations
    Adaptive,
    /// Tier 2: Goal-directed — parametric exploration within bounds
    Intentional,
    /// Tier 3: Self-reflective — creative CSG combinations, novel geometry
    Creative,
    /// Tier 4: Meta-cognitive — cross-domain inspiration, surprise-driven design
    Transcendent,
}

impl CivicTier {
    /// Classify a consciousness score [0.0, 1.0] into a tier
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s < 0.2 => Self::Reflexive,
            s if s < 0.4 => Self::Adaptive,
            s if s < 0.6 => Self::Intentional,
            s if s < 0.8 => Self::Creative,
            _ => Self::Transcendent,
        }
    }

    /// Maximum CSG tree depth allowed at this tier
    pub fn max_csg_depth(&self) -> usize {
        match self {
            Self::Reflexive => 1,    // Single primitive
            Self::Adaptive => 2,     // One boolean op
            Self::Intentional => 4,  // Nested operations
            Self::Creative => 8,     // Complex assemblies
            Self::Transcendent => 16, // Unrestricted
        }
    }

    /// Number of design variants to explore
    pub fn exploration_width(&self) -> usize {
        match self {
            Self::Reflexive => 1,
            Self::Adaptive => 2,
            Self::Intentional => 4,
            Self::Creative => 8,
            Self::Transcendent => 16,
        }
    }

    /// Similarity threshold for search — lower = wider net
    pub fn search_threshold(&self) -> f32 {
        match self {
            Self::Reflexive => 0.9,   // Very strict matching
            Self::Adaptive => 0.8,
            Self::Intentional => 0.7,
            Self::Creative => 0.5,    // Cross-domain inspiration
            Self::Transcendent => 0.3, // Maximum openness
        }
    }

    /// Confidence penalty — lower tiers require higher confidence to proceed
    pub fn min_confidence(&self) -> f32 {
        match self {
            Self::Reflexive => 0.9,
            Self::Adaptive => 0.7,
            Self::Intentional => 0.5,
            Self::Creative => 0.3,
            Self::Transcendent => 0.1,
        }
    }
}

/// Fabrication parameters adjusted by consciousness level
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsciousnessGatedParams {
    pub tier: String,
    pub consciousness_score: f32,
    pub max_csg_depth: usize,
    pub exploration_width: usize,
    pub search_threshold: f32,
    pub min_confidence: f32,
    pub creative_features_enabled: bool,
}

/// Compute consciousness-gated fabrication parameters from a score.
/// This is called by the LUCID bridge to adjust fabrication behaviour
/// based on Symthaea's current consciousness state.
fn consciousness_gated_params(consciousness_score: f32) -> ConsciousnessGatedParams {
    let score = consciousness_score.clamp(0.0, 1.0);
    let tier = CivicTier::from_score(score);
    ConsciousnessGatedParams {
        tier: format!("{:?}", tier),
        consciousness_score: score,
        max_csg_depth: tier.max_csg_depth(),
        exploration_width: tier.exploration_width(),
        search_threshold: tier.search_threshold(),
        min_confidence: tier.min_confidence(),
        creative_features_enabled: matches!(tier, CivicTier::Creative | CivicTier::Transcendent),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConsciousnessGatedInput {
    pub base_design_hash: ActionHash,
    pub intent_modifiers: Vec<SerializedBinding>,
    pub material_constraints: Vec<String>,
    pub printer_constraints: Option<String>,
    /// Consciousness score from Symthaea cognitive loop [0.0, 1.0].
    /// Higher values enable more creative/exploratory generation.
    pub consciousness_score: f32,
}

/// Generate a parametric variant with consciousness-gated exploration depth.
/// At low consciousness, sticks to proven templates with minimal variation.
/// At high consciousness, explores novel CSG combinations and wider search.
#[hdk_extern]
pub fn generate_consciousness_gated_variant(
    input: ConsciousnessGatedInput,
) -> ExternResult<Record> {
    rate_limit_caller()?;
    let params = consciousness_gated_params(input.consciousness_score);

    // Gate: if confidence would be below tier minimum, fall back to template
    let recognized = input.intent_modifiers.iter()
        .filter(|b| matches!(b.role.as_str(), "shape" | "dimension" | "material" | "feature" | "transform"))
        .count();
    let raw_confidence = if input.intent_modifiers.is_empty() {
        0.5
    } else {
        0.5 + 0.5 * (recognized as f32 / input.intent_modifiers.len() as f32)
    };

    // At low consciousness, require higher confidence to proceed with parametric
    let use_parametric = raw_confidence >= params.min_confidence;

    let modifiers = if use_parametric {
        input.intent_modifiers.clone()
    } else {
        // Fall back to just the base shape binding
        input.intent_modifiers.iter()
            .filter(|b| b.role == "shape")
            .cloned()
            .collect::<Vec<_>>()
    };

    let result = generate_parametric_variant(GenerateVariantInput {
        base_design_hash: input.base_design_hash,
        intent_modifiers: modifiers,
        material_constraints: input.material_constraints,
        printer_constraints: input.printer_constraints,
    })?;

    let _ = emit_signal(&TypedFabricationSignal {
        domain: FabricationDomain::Symthaea,
        event_type: FabricationEventType::ConsciousnessGatedVariant,
        payload: serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
    });

    Ok(result)
}

/// Query consciousness-gated parameters (read-only, for UI display)
#[hdk_extern]
pub fn get_consciousness_params(consciousness_score: f32) -> ExternResult<ConsciousnessGatedParams> {
    Ok(consciousness_gated_params(consciousness_score))
}

// =============================================================================
// UNIT TESTS (pure functions only -- HDK externs require conductor)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // generate_vector_hash
    // =========================================================================

    #[test]
    fn test_generate_vector_hash_deterministic() {
        let encoder = FabTextEncoder::new(FAB_HDC_DIM);
        let hv = encoder.encode("a weatherproof bracket for 12mm pipe");
        let hash1 = generate_vector_hash(&hv);
        let hash2 = generate_vector_hash(&hv);
        assert_eq!(hash1, hash2, "Same vector must produce same hash");
        assert!(hash1.starts_with("hdc_"), "Hash must start with hdc_ prefix");
    }

    #[test]
    fn test_generate_vector_hash_different() {
        let encoder = FabTextEncoder::new(FAB_HDC_DIM);
        let hv_a = encoder.encode("bracket for pipe");
        let hv_b = encoder.encode("gear for motor");
        let hash_a = generate_vector_hash(&hv_a);
        let hash_b = generate_vector_hash(&hv_b);
        assert_ne!(hash_a, hash_b, "Different vectors must produce different hashes");
    }

    // =========================================================================
    // parse_semantic_bindings
    // =========================================================================

    #[test]
    fn test_parse_semantic_bindings_bracket() {
        let bindings = parse_semantic_bindings("I need a bracket");
        assert!(bindings.iter().any(|b| b.concept == "bracket" && b.role == "Base"));
    }

    #[test]
    fn test_parse_semantic_bindings_dimensional() {
        let bindings = parse_semantic_bindings("pipe is 12mm diameter");
        assert!(bindings.iter().any(|b| b.concept.contains("12mm") && b.role == "Dimensional"));
    }

    #[test]
    fn test_parse_semantic_bindings_material() {
        let bindings = parse_semantic_bindings("print it in pla");
        assert!(bindings.iter().any(|b| b.concept == "pla" && b.role == "Material"));
    }

    #[test]
    fn test_parse_semantic_bindings_empty() {
        let bindings = parse_semantic_bindings("just a thing");
        // "just a thing" contains no recognized keywords
        assert!(bindings.is_empty(), "Unrecognized description should yield no bindings");
    }

    // =========================================================================
    // lookup_mtbf
    // =========================================================================

    #[test]
    fn test_mtbf_fdm_printer() {
        assert_eq!(lookup_mtbf("fdm_printer"), 3000);
        assert_eq!(lookup_mtbf("FDM_PRINTER"), 3000); // Case insensitive
    }

    #[test]
    fn test_mtbf_drill() {
        assert_eq!(lookup_mtbf("power_drill"), 1500);
        assert_eq!(lookup_mtbf("drill"), 1500);
    }

    #[test]
    fn test_mtbf_general_fallback() {
        assert_eq!(lookup_mtbf("unknown_device"), 2000);
        assert_eq!(lookup_mtbf("general"), 2000);
    }

    // =========================================================================
    // analyze_sensor_degradation
    // =========================================================================

    #[test]
    fn test_analyze_sensor_degradation_empty() {
        let analysis = analyze_sensor_degradation(&[]);
        assert!((analysis.degradation_score - 0.5).abs() < f32::EPSILON,
            "Empty readings should return default 0.5 degradation score");
        assert_eq!(analysis.likely_component, "unknown");
    }

    #[test]
    fn test_analyze_sensor_degradation_vibration_dominant() {
        let readings = vec![
            SensorReading { timestamp: 1000, sensor_type: "vibration".into(), value: 10.0, unit: "mm/s".into() },
            SensorReading { timestamp: 2000, sensor_type: "vibration".into(), value: 12.0, unit: "mm/s".into() },
            SensorReading { timestamp: 3000, sensor_type: "temperature".into(), value: 26.0, unit: "C".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert_eq!(analysis.likely_component, "bearing",
            "Vibration-dominant readings should point to bearing");
        assert!(analysis.summary.vibration_rms > 0.0);
    }

    #[test]
    fn test_analyze_sensor_degradation_temp_dominant() {
        let readings = vec![
            SensorReading { timestamp: 1000, sensor_type: "temperature".into(), value: 80.0, unit: "C".into() },
            SensorReading { timestamp: 2000, sensor_type: "temperature".into(), value: 85.0, unit: "C".into() },
            SensorReading { timestamp: 3000, sensor_type: "vibration".into(), value: 0.5, unit: "mm/s".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert_eq!(analysis.likely_component, "thermal_component",
            "Temperature-dominant readings should point to thermal_component");
        assert!(analysis.summary.temp_max >= 85.0);
    }

    #[test]
    fn test_analyze_sensor_torque_variance() {
        let readings = vec![
            SensorReading { timestamp: 1000, sensor_type: "torque".into(), value: 5.0, unit: "Nm".into() },
            SensorReading { timestamp: 2000, sensor_type: "torque".into(), value: 8.0, unit: "Nm".into() },
            SensorReading { timestamp: 3000, sensor_type: "torque".into(), value: 3.0, unit: "Nm".into() },
            SensorReading { timestamp: 4000, sensor_type: "torque".into(), value: 9.0, unit: "Nm".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert_eq!(analysis.likely_component, "chuck_mechanism");
        assert!(analysis.summary.torque_variance > 0.0);
    }

    #[test]
    fn test_degradation_score_healthy() {
        // Low-vibration, normal temperature → low score
        let readings = vec![
            SensorReading { timestamp: 1000, sensor_type: "vibration".into(), value: 0.5, unit: "mm/s".into() },
            SensorReading { timestamp: 2000, sensor_type: "temperature".into(), value: 30.0, unit: "C".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert!(analysis.degradation_score < 0.5, "Healthy readings should have low score, got {}", analysis.degradation_score);
    }

    #[test]
    fn test_degradation_score_critical() {
        // High vibration + high temperature → high score
        let readings = vec![
            SensorReading { timestamp: 1000, sensor_type: "vibration".into(), value: 15.0, unit: "mm/s".into() },
            SensorReading { timestamp: 2000, sensor_type: "vibration".into(), value: 18.0, unit: "mm/s".into() },
            SensorReading { timestamp: 3000, sensor_type: "temperature".into(), value: 95.0, unit: "C".into() },
            SensorReading { timestamp: 4000, sensor_type: "temperature".into(), value: 100.0, unit: "C".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert!(analysis.degradation_score > 0.5, "Critical readings should have high score, got {}", analysis.degradation_score);
    }

    // =========================================================================
    // numeric helpers
    // =========================================================================

    #[test]
    fn test_rms() {
        assert!((rms(&[3.0, 4.0]) - 3.5355).abs() < 0.01);
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn test_linear_slope_positive() {
        // Perfect positive slope: y = 2x
        let pairs = vec![(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)];
        assert!((linear_slope(&pairs) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_slope_flat() {
        let pairs = vec![(0.0, 5.0), (1.0, 5.0), (2.0, 5.0)];
        assert!((linear_slope(&pairs)).abs() < 0.01);
    }

    #[test]
    fn test_variance_uniform() {
        assert!((variance(&[5.0, 5.0, 5.0])).abs() < 0.01);
    }

    #[test]
    fn test_variance_spread() {
        assert!(variance(&[0.0, 10.0]) > 0.0);
    }

    // =========================================================================
    // calculate_local_adjustments
    // =========================================================================

    #[test]
    fn test_calculate_local_adjustments_solar() {
        let adjustments = calculate_local_adjustments(&[], &[], "solar");
        assert_eq!(adjustments.get("print_speed").map(|s| s.as_str()), Some("reduced_10%"));
        assert_eq!(adjustments.get("infill_pattern").map(|s| s.as_str()), Some("efficient"));
        assert_eq!(adjustments.get("local_optimized").map(|s| s.as_str()), Some("true"));
    }

    #[test]
    fn test_calculate_local_adjustments_grid() {
        let adjustments = calculate_local_adjustments(&[], &[], "grid");
        assert_eq!(adjustments.get("print_time").map(|s| s.as_str()), Some("off_peak"));
        assert_eq!(adjustments.get("local_optimized").map(|s| s.as_str()), Some("true"));
        // solar-specific keys should not be present
        assert!(adjustments.get("print_speed").is_none());
    }

    // =========================================================================
    // calculate_improvement_metrics
    // =========================================================================

    #[test]
    fn test_calculate_improvement_metrics() {
        let mut adjustments = std::collections::HashMap::new();
        adjustments.insert("print_time".to_string(), "off_peak".to_string());

        let metrics = calculate_improvement_metrics(&adjustments);
        assert!(metrics.contains_key("energy_efficiency"), "Must contain energy_efficiency");
        assert!(metrics.contains_key("material_utilization"), "Must contain material_utilization");
        assert!(metrics.contains_key("local_economy_boost"), "Must contain local_economy_boost");
        assert!(metrics.contains_key("grid_friendliness"),
            "Must contain grid_friendliness when print_time is present");
    }

    // =========================================================================
    // build_parametric_config
    // =========================================================================

    #[test]
    fn test_build_parametric_config_default_cube() {
        let config = build_parametric_config(&[]);
        let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
        assert_eq!(parsed["kernel"], "symthaea-fabrication-kernel");
        assert_eq!(parsed["csg_tree"]["Primitive"], "Cube");
    }

    #[test]
    fn test_build_parametric_config_cylinder_shape() {
        let modifiers = vec![SerializedBinding {
            concept: "cylinder tube".to_string(),
            role: "shape".to_string(),
            weight: 1.0,
        }];
        let config = build_parametric_config(&modifiers);
        let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
        assert_eq!(parsed["csg_tree"]["Primitive"], "Cylinder");
    }

    #[test]
    fn test_build_parametric_config_with_dimensions() {
        let modifiers = vec![
            SerializedBinding { concept: "bracket".to_string(), role: "shape".to_string(), weight: 1.0 },
            SerializedBinding { concept: "50x30x10".to_string(), role: "dimension".to_string(), weight: 1.0 },
        ];
        let config = build_parametric_config(&modifiers);
        let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
        // Should have a Transform wrapping the primitive
        assert!(parsed["csg_tree"]["Transform"].is_object(), "Should have transform for dimensions");
        let scale = &parsed["csg_tree"]["Transform"]["transform"]["scale"];
        assert!((scale[0].as_f64().unwrap() - 0.05).abs() < 0.001);
    }

    // =========================================================================
    // extract_primary_category
    // =========================================================================

    #[test]
    fn test_extract_category_fasteners() {
        let bindings = vec![SerializedBinding { concept: "bracket".to_string(), role: "Base".to_string(), weight: 1.0 }];
        assert_eq!(extract_primary_category(&bindings), "fasteners");
    }

    #[test]
    fn test_extract_category_mechanical() {
        let bindings = vec![SerializedBinding { concept: "gear".to_string(), role: "Base".to_string(), weight: 1.0 }];
        assert_eq!(extract_primary_category(&bindings), "mechanical");
    }

    #[test]
    fn test_extract_category_general() {
        let bindings = vec![SerializedBinding { concept: "widget".to_string(), role: "Base".to_string(), weight: 1.0 }];
        assert_eq!(extract_primary_category(&bindings), "general");
    }

    #[test]
    fn test_extract_category_empty() {
        assert_eq!(extract_primary_category(&[]), "general");
    }

    // =========================================================================
    // consciousness gating
    // =========================================================================

    #[test]
    fn test_consciousness_tier_reflexive() {
        let tier = CivicTier::from_score(0.1);
        assert_eq!(tier, CivicTier::Reflexive);
        assert_eq!(tier.max_csg_depth(), 1);
        assert_eq!(tier.exploration_width(), 1);
        assert!(tier.search_threshold() > 0.85);
    }

    #[test]
    fn test_consciousness_tier_creative() {
        let tier = CivicTier::from_score(0.75);
        assert_eq!(tier, CivicTier::Creative);
        assert_eq!(tier.max_csg_depth(), 8);
        assert_eq!(tier.exploration_width(), 8);
        assert!(tier.search_threshold() < 0.6);
    }

    #[test]
    fn test_consciousness_tier_transcendent() {
        let tier = CivicTier::from_score(0.95);
        assert_eq!(tier, CivicTier::Transcendent);
        assert_eq!(tier.max_csg_depth(), 16);
        assert!(tier.min_confidence() < 0.2, "Transcendent should accept low confidence");
    }

    #[test]
    fn test_consciousness_gated_params_low() {
        let params = consciousness_gated_params(0.1);
        assert_eq!(params.tier, "Reflexive");
        assert!(!params.creative_features_enabled);
        assert_eq!(params.exploration_width, 1);
    }

    #[test]
    fn test_consciousness_gated_params_high() {
        let params = consciousness_gated_params(0.9);
        assert_eq!(params.tier, "Transcendent");
        assert!(params.creative_features_enabled);
        assert_eq!(params.exploration_width, 16);
    }

    #[test]
    fn test_consciousness_gated_params_clamped() {
        let low = consciousness_gated_params(-1.0);
        assert_eq!(low.tier, "Reflexive");
        let high = consciousness_gated_params(2.0);
        assert_eq!(high.tier, "Transcendent");
    }

    #[test]
    fn test_consciousness_tiers_monotonic() {
        // As consciousness increases, exploration should increase
        let scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let widths: Vec<usize> = scores.iter().map(|s| {
            CivicTier::from_score(*s).exploration_width()
        }).collect();
        for w in widths.windows(2) {
            assert!(w[1] >= w[0], "exploration_width should be monotonically non-decreasing");
        }
    }

    #[test]
    fn test_build_parametric_config_with_hole_feature() {
        let modifiers = vec![
            SerializedBinding { concept: "mounting hole".to_string(), role: "feature".to_string(), weight: 1.0 },
        ];
        let config = build_parametric_config(&modifiers);
        let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
        assert_eq!(parsed["csg_tree"]["Boolean"]["op"], "Subtract",
            "Hole feature should create a Boolean Subtract");
    }

    // =========================================================================
    // compute_binding_overlap tests
    // =========================================================================

    #[test]
    fn test_binding_overlap_exact_match() {
        let a = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.8 },
        ];
        let b = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.5 },
        ];
        let result = compute_binding_overlap(&a, &b);
        assert_eq!(result, vec!["cup"]);
    }

    #[test]
    fn test_binding_overlap_different_roles_no_match() {
        let a = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.8 },
        ];
        let b = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Modifier".to_string(), weight: 0.5 },
        ];
        let result = compute_binding_overlap(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_binding_overlap_different_concepts_no_match() {
        let a = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.8 },
        ];
        let b = vec![
            SerializedBinding { concept: "plate".to_string(), role: "Base".to_string(), weight: 0.5 },
        ];
        let result = compute_binding_overlap(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_binding_overlap_multiple_matches() {
        let a = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.8 },
            SerializedBinding { concept: "handle".to_string(), role: "Modifier".to_string(), weight: 0.5 },
        ];
        let b = vec![
            SerializedBinding { concept: "cup".to_string(), role: "Base".to_string(), weight: 0.6 },
            SerializedBinding { concept: "handle".to_string(), role: "Modifier".to_string(), weight: 0.3 },
            SerializedBinding { concept: "lid".to_string(), role: "Modifier".to_string(), weight: 0.2 },
        ];
        let result = compute_binding_overlap(&a, &b);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"cup".to_string()));
        assert!(result.contains(&"handle".to_string()));
    }

    #[test]
    fn test_binding_overlap_empty_inputs() {
        let result = compute_binding_overlap(&[], &[]);
        assert!(result.is_empty());
    }
}
