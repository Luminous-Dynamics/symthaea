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
use fabrication_common::FabricationSignal;

// HDC Configuration
const HDC_DIMENSIONS: u32 = 4_096;
const SIMILARITY_THRESHOLD: f32 = 0.7;

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
        vector_dimensions: HDC_DIMENSIONS,
        vector_hash: vector_hash.clone(),
        semantic_bindings: bindings.clone(),
        generation_method: "symthaea_hdc".to_string(),
        language: input.language.unwrap_or_else(|| "en".to_string()),
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::HdcIntent(intent))?;

    let _ = emit_signal(&FabricationSignal {
        event_type: "intent_generated".to_string(),
        source_zome: "symthaea".to_string(),
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(author, hash.clone(), LinkTypes::AuthorToIntents, ())?;

    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))?;

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
    // Get base intent
    let base_record = get(input.base_intent_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Base intent not found".into())))?;

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
        vector_dimensions: HDC_DIMENSIONS,
        vector_hash: vector_hash.clone(),
        semantic_bindings: all_bindings.clone(),
        generation_method: "lateral_binding".to_string(),
        language: base_intent.language,
        author: author.clone(),
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::HdcIntent(combined_intent))?;

    let _ = emit_signal(&FabricationSignal {
        event_type: "intent_bound".to_string(),
        source_zome: "symthaea".to_string(),
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(author, hash.clone(), LinkTypes::AuthorToIntents, ())?;
    create_link(input.base_intent_hash, hash.clone(), LinkTypes::IntentToDesigns, ())?;

    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))?;

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
    let threshold = input.threshold.unwrap_or(SIMILARITY_THRESHOLD);
    let limit = input.limit.unwrap_or(10) as usize;

    // Get query intent
    let query_record = get(input.intent_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Intent not found".into())))?;

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
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AuthorToIntents)?, GetStrategy::default())?;

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
                        // Record the match
                        let match_entry = SemanticMatchEntry {
                            query_intent_hash: input.intent_hash.clone(),
                            matched_design_hash: hash.clone(),
                            similarity_score: similarity,
                            matched_bindings: matched.clone(),
                            searched_at: Timestamp::from_micros(now.as_micros() as i64),
                        };
                        let _ = create_entry(EntryTypes::SemanticMatch(match_entry));

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

    // Create the generated design
    let generated = GeneratedDesignEntry {
        intent_hash: intent_result.record.action_address().clone(),
        base_design_hash: Some(input.base_design_hash.clone()),
        parametric_config: serde_json::to_string(&input.intent_modifiers)
            .unwrap_or_else(|_| "{}".to_string()),
        material_constraints: input.material_constraints,
        printer_constraints: input.printer_constraints,
        generated_file_cid: None, // Would be populated after actual generation
        confidence_score: 0.8,    // Placeholder
        generation_time_ms: 0,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
    };

    let hash = create_entry(EntryTypes::GeneratedDesign(generated))?;

    let _ = emit_signal(&FabricationSignal {
        event_type: "variant_generated".to_string(),
        source_zome: "symthaea".to_string(),
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(
        input.base_design_hash,
        hash.clone(),
        LinkTypes::IntentToDesigns,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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

    let _ = emit_signal(&FabricationSignal {
        event_type: "optimization_completed".to_string(),
        source_zome: "symthaea".to_string(),
        payload: format!(r#"{{"hash":"{}"}}"#, hash),
    });

    create_link(
        input.design_hash,
        hash.clone(),
        LinkTypes::DesignToOptimizations,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
// REPAIR PREDICTION
// =============================================================================

/// Default mean time between failures in hours, used when `mtbf_hours` is not
/// provided by the caller. Callers should supply asset-specific MTBF data when
/// available; this default is a conservative baseline.
const DEFAULT_MTBF_HOURS: u32 = 2000;

#[derive(Serialize, Deserialize, Debug)]
pub struct PredictRepairInput {
    pub property_asset_hash: ActionHash,
    pub sensor_history: Vec<SensorReading>,
    pub usage_hours: u32,
    /// Asset-specific mean time between failures (hours). Falls back to
    /// [`DEFAULT_MTBF_HOURS`] (2000) when `None`.
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
    pub matching_repair_designs: Vec<ActionHash>,
}

/// Predict repair needs from digital twin data
#[hdk_extern]
pub fn predict_repair_needs(input: PredictRepairInput) -> ExternResult<RepairPredictionResult> {
    // Analyze sensor history for degradation patterns
    let analysis = analyze_sensor_degradation(&input.sensor_history);

    // Calculate failure probability based on usage
    let base_mtbf = input.mtbf_hours.unwrap_or(DEFAULT_MTBF_HOURS);
    let usage_factor = input.usage_hours as f32 / base_mtbf as f32;
    let sensor_factor = analysis.degradation_rate;

    let failure_probability = (usage_factor * sensor_factor).min(1.0);

    // Estimate remaining useful life
    let remaining_hours = if failure_probability < 0.9 {
        ((1.0 - failure_probability) * base_mtbf as f32) as u32
    } else {
        0
    };

    // Determine recommended action
    let recommended_action = if failure_probability > 0.8 {
        "PrintReplacement".to_string()
    } else if failure_probability > 0.6 {
        "ScheduleReplacement".to_string()
    } else if failure_probability > 0.4 {
        "OrderMaterials".to_string()
    } else {
        "Monitor".to_string()
    };

    // Search for matching repair designs (would query designs zome)
    let matching_designs = Vec::new(); // Placeholder

    Ok(RepairPredictionResult {
        predicted_component: analysis.likely_component,
        failure_probability,
        estimated_remaining_hours: remaining_hours,
        recommended_action,
        matching_repair_designs: matching_designs,
    })
}

struct DegradationAnalysis {
    degradation_rate: f32,
    likely_component: String,
}

fn analyze_sensor_degradation(readings: &[SensorReading]) -> DegradationAnalysis {
    if readings.is_empty() {
        return DegradationAnalysis {
            degradation_rate: 0.5,
            likely_component: "unknown".to_string(),
        };
    }

    // Simple trend analysis
    let mut vibration_trend = 0.0;
    let mut temp_trend = 0.0;

    for reading in readings {
        match reading.sensor_type.as_str() {
            "vibration" => vibration_trend += reading.value * 0.1,
            "temperature" => temp_trend += (reading.value - 25.0).abs() * 0.01,
            _ => {}
        }
    }

    let degradation_rate = ((vibration_trend + temp_trend) / readings.len() as f32).min(2.0);

    let likely_component = if vibration_trend > temp_trend {
        "bearing".to_string()
    } else {
        "thermal_component".to_string()
    };

    DegradationAnalysis {
        degradation_rate,
        likely_component,
    }
}

// =============================================================================
// QUERIES
// =============================================================================

#[hdk_extern]
pub fn get_my_intents(_: ()) -> ExternResult<Vec<Record>> {
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
    Ok(results)
}

#[hdk_extern]
pub fn get_design_optimizations(design_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(design_hash, LinkTypes::DesignToOptimizations)?, GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                results.push(record);
            }
        }
    }
    Ok(results)
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
    // analyze_sensor_degradation
    // =========================================================================

    #[test]
    fn test_analyze_sensor_degradation_empty() {
        let analysis = analyze_sensor_degradation(&[]);
        assert!((analysis.degradation_rate - 0.5).abs() < f32::EPSILON,
            "Empty readings should return default 0.5 degradation rate");
        assert_eq!(analysis.likely_component, "unknown");
    }

    #[test]
    fn test_analyze_sensor_degradation_vibration_dominant() {
        let readings = vec![
            SensorReading { timestamp: 1, sensor_type: "vibration".into(), value: 10.0, unit: "mm/s".into() },
            SensorReading { timestamp: 2, sensor_type: "vibration".into(), value: 12.0, unit: "mm/s".into() },
            SensorReading { timestamp: 3, sensor_type: "temperature".into(), value: 26.0, unit: "C".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert_eq!(analysis.likely_component, "bearing",
            "Vibration-dominant readings should point to bearing");
    }

    #[test]
    fn test_analyze_sensor_degradation_temp_dominant() {
        let readings = vec![
            SensorReading { timestamp: 1, sensor_type: "temperature".into(), value: 80.0, unit: "C".into() },
            SensorReading { timestamp: 2, sensor_type: "temperature".into(), value: 85.0, unit: "C".into() },
            SensorReading { timestamp: 3, sensor_type: "vibration".into(), value: 0.5, unit: "mm/s".into() },
        ];
        let analysis = analyze_sensor_degradation(&readings);
        assert_eq!(analysis.likely_component, "thermal_component",
            "Temperature-dominant readings should point to thermal_component");
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
}
