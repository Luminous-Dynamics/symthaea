# Fabrication hApp API Reference

> Generated from source on 2026-03-01 | **76 `#[hdk_extern]` functions** across 7 coordinator zomes

This document is auto-extracted from the actual Rust coordinator source files. Every function listed below has a corresponding `#[hdk_extern]` attribute in the codebase.

---

## Table of Contents

| Zome | Functions | Description |
|------|-----------|-------------|
| [designs](#designs-zome) | 15 | Design CRUD, versioning, forking, discovery, parametric generation |
| [printers](#printers-zome) | 12 | Printer registration, matching, availability, compatibility |
| [prints](#prints-zome) | 16 | Print job lifecycle, PoGF scoring, Cincinnati monitoring, statistics |
| [materials](#materials-zome) | 6 | Material CRUD and discovery |
| [verification](#verification-zome) | 6 | Safety verification, epistemic scoring, claims |
| [bridge](#bridge-zome) | 9 | Anticipatory repair, marketplace, supply chain, audit trail |
| [symthaea](#symthaea-zome) | 12 | HDC encoding, semantic search, parametric generation, consciousness gating |

Pagination note: Functions marked with (paginated) accept an optional `PaginationInput { offset: u32, limit: u32 }` field and return `PaginatedResponse<T> { items, total, offset, limit }`. Limit is clamped to max 100.

---

## Designs Zome

**Source**: `zomes/designs/coordinator/src/lib.rs`

### CRUD Operations

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `create_design` | `CreateDesignInput { title, description, category: DesignCategory, intent_vector?: HdcHypervector, parametric_schema?: ParametricSchema, constraint_graph?: ConstraintGraph, material_compatibility: Vec<MaterialBinding>, circularity_score: f32, embodied_energy_kwh: f32, repair_manifest?: RepairManifest, license: License, safety_class: SafetyClass }` | `Record` | Create a new design with full metadata. Links to author, category anchor, and all-designs anchor. |
| `get_design` | `ActionHash` | `Option<Record>` | Retrieve a design by its action hash. |
| `update_design` | `UpdateDesignInput { original_action_hash: ActionHash, title?, description?, category?: DesignCategory, intent_vector?, parametric_schema?, constraint_graph?, material_compatibility?, circularity_score?: f32, embodied_energy_kwh?: f32, repair_manifest?, license?, safety_class?, epistemic?: DesignEpistemic }` | `Record` | Update an existing design. Author-only. All fields optional except `original_action_hash`. |
| `delete_design` | `ActionHash` | `ActionHash` | Soft-delete a design. Author-only. Returns deletion action hash. |

### File Management

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `add_design_file` | `AddFileInput { design_hash: ActionHash, file: DesignFile }` | `Record` | Attach a file (STL, 3MF, STEP, etc.) to a design. Author-only. |
| `get_design_files` | `HashListInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all files attached to a design. (paginated) |

### Versioning & Forking

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `fork_design` | `ForkDesignInput { parent_hash: ActionHash, modification_notes: String, title?, description?, intent_modifications?: Vec<SemanticBinding> }` | `Record` | Create a derivative of an existing design. Respects license (blocks Proprietary and ND variants). Inherits parent metadata. |
| `get_design_history` | `HashListInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all update versions of a design in chronological order. (paginated) |
| `get_design_forks` | `HashListInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all designs forked from this parent. (paginated) |

### Discovery

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_designs_by_author` | `GetDesignsByAuthorInput { author: AgentPubKey, pagination? }` | `PaginatedResponse<Record>` | Get all designs created by a specific agent. (paginated) |
| `get_designs_by_category` | `GetDesignsByCategoryInput { category: DesignCategory, pagination? }` | `PaginatedResponse<Record>` | Get all designs in a category. (paginated) |
| `search_designs` | `DesignSearchQuery { query?: String, category?: DesignCategory, safety_class?: SafetyClass, min_circularity?: f32, license?: License, limit?: u32, pagination? }` | `PaginatedResponse<Record>` | Full-text search with multi-field filtering. Searches title and description. (paginated) |
| `get_featured_designs` | `GetFeaturedDesignsInput { pagination? }` | `PaginatedResponse<Record>` | Get featured/highlighted designs. (paginated) |

### Parametric Operations

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_parameters` | `ActionHash` | `Option<ParametricSchema>` | Get the parametric schema for a design. |
| `generate_variant` | `GenerateVariantInput { design_hash: ActionHash, parameters: HashMap<String, ParameterValue> }` | `GeneratedVariant { design_hash, parameters_used, output_file?, generation_status, csg_record_hash? }` | Generate a parametric variant. Delegates to the symthaea zome via cross-zome call for CSG generation. |

---

## Printers Zome

**Source**: `zomes/printers/coordinator/src/lib.rs`

### CRUD Operations

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `register_printer` | `RegisterPrinterInput { name: String, location?: GeoLocation, printer_type: PrinterType, capabilities: PrinterCapabilities, materials_available: Vec<MaterialType>, rates?: PrinterRates }` | `Record` | Register a new 3D printer. Links to owner, type anchor, and geohash anchor. |
| `get_printer` | `ActionHash` | `Option<Record>` | Retrieve a printer by its action hash. |
| `update_printer` | `UpdatePrinterInput { original_action_hash: ActionHash, name?, location?: GeoLocation, capabilities?: PrinterCapabilities, materials_available?: Vec<MaterialType>, rates?: PrinterRates }` | `Record` | Update printer information. Owner-only. |
| `deactivate_printer` | `ActionHash` | `ActionHash` | Deactivate a printer (sets status to Deactivated). Owner-only. Returns deletion action hash. |

### Discovery & Matching

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_my_printers` | `MyPrintersInput { pagination? }` | `PaginatedResponse<Record>` | Get all printers owned by the calling agent. (paginated) |
| `find_printers_nearby` | `FindNearbyInput { location: GeoLocation, radius_km: u32, pagination? }` | `PaginatedResponse<PrinterMatch>` | Find printers near a location using geohash prefix search. Returns matches with distance and compatibility scores. (paginated) |
| `find_printers_by_capability` | `FindByCapabilityInput { requirements: PrinterRequirements, pagination? }` | `PaginatedResponse<PrinterMatch>` | Find printers matching capability requirements (build volume, material, type, layer height, bed, enclosure, hotend temp). (paginated) |
| `get_available_printers` | `GetAvailablePrintersInput { pagination? }` | `PaginatedResponse<Record>` | Get all printers with Available status. (paginated) |
| `match_design_to_printers` | `MatchDesignInput { design_hash: ActionHash, location?: GeoLocation, limit?: u32 }` | `Vec<PrinterMatch>` | Intelligent matching: fetches design requirements, scores all available printers on material, volume, capability, and distance. |
| `check_printer_compatibility` | `CheckCompatibilityInput { printer_hash: ActionHash, design_hash: ActionHash }` | `CompatibilityResult { compatible: bool, score: f32, issues: Vec<String>, recommendations: Vec<String> }` | Detailed compatibility check between a specific printer and design. |

### Status Management

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `update_availability` | `UpdateAvailabilityInput { printer_hash: ActionHash, status: AvailabilityStatus, message?: String, eta_available?: u32, current_job?: ActionHash }` | `Record` | Update printer availability status. Owner-only. |
| `get_printer_queue` | `ActionHash` | `Vec<Record>` | Get print jobs queued for a printer (cross-zome query). |

---

## Prints Zome

**Source**: `zomes/prints/coordinator/src/lib.rs`

Implements print job lifecycle, Proof of Grounded Fabrication (PoGF) scoring, Cincinnati Algorithm quality monitoring, and MYCELIUM (CIV) reputation integration.

### Job Lifecycle

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `create_print_job` | `CreatePrintJobInput { design_hash: ActionHash, printer_hash: ActionHash, settings: PrintSettings, energy_source?: EnergyType, material_passport?: MaterialPassport }` | `Record` | Create a new print job. Enforces safety class verification for Class3+ designs via cross-zome call to verification zome. |
| `accept_print_job` | `ActionHash` | `Record` | Accept a pending print job. Printer-owner-only. Transitions status Pending -> Accepted. |
| `start_print` | `ActionHash` | `Record` | Start printing an accepted job. Printer-owner-only. Transitions status Accepted -> Printing. |
| `update_print_progress` | `UpdateProgressInput { job_hash: ActionHash, progress_percent: u8, current_layer?: u32, material_used_grams?: f32 }` | `Record` | Update print progress. Printer-owner-only. |
| `complete_print` | `CompletePrintInput { job_hash: ActionHash, result: PrintResult }` | `Record` | Complete a print job with result status. Printer-owner-only. Transitions status Printing -> Completed. |
| `cancel_print` | `CancelPrintInput { job_hash: ActionHash, reason: String }` | `Record` | Cancel a print job. Requester-or-owner. Transitions status to Cancelled. |

### Print Records & Quality

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `record_print_result` | `RecordPrintInput { job_hash: ActionHash, result: PrintResult, energy_used_kwh: f32, photos: Vec<String>, notes: String, issues: Vec<PrintIssue>, dimensional_measurements?: Vec<DimensionalMeasurement>, attestations?: PogfAttestationBundle }` | `Record` | Record detailed print results with PoGF scoring, quality assessment, and optional attestation chain. Calculates MYCELIUM reputation reward. |
| `get_print_record` | `ActionHash` (job_hash) | `Option<Record>` | Get the print result record for a completed job. |
| `add_print_photos` | `AddPhotosInput { record_hash: ActionHash, photos: Vec<String> }` | `Record` | Add photos to an existing print record. |

### Cincinnati Algorithm (Quality Monitoring)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `start_cincinnati_monitoring` | `StartCincinnatiInput { job_hash: ActionHash, total_layers: u32, sampling_rate_hz: u32 }` | `Record` | Start a Cincinnati Algorithm monitoring session for a print job. Printer-owner-only. |
| `report_cincinnati_anomaly` | `ReportAnomalyInput { session_id: String, anomaly: AnomalyEvent }` | `Record` | Report a quality anomaly detected during monitoring. Printer-owner-only. |
| `update_cincinnati_session` | `UpdateCincinnatiInput { session_hash: ActionHash, current_layer: u32, health_score: f32, anomaly_count: u32 }` | `Record` | Update Cincinnati session progress with current health score. |

### Queries

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_my_print_jobs` | `AgentPaginationInput { pagination? }` | `PaginatedResponse<Record>` | Get all print jobs requested by the calling agent. (paginated) |
| `get_printer_jobs` | `HashPaginationInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all print jobs for a specific printer. (paginated) |
| `get_design_prints` | `HashPaginationInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all print records for a design. (paginated) |
| `get_print_statistics` | `ActionHash` (design_hash) | `DesignPrintStats { design_hash, total_prints, successful_prints, failed_prints, average_quality: f32, average_pog_score: f32, total_mycelium_earned: u64, common_issues }` | Get aggregated print statistics for a design. |

---

## Materials Zome

**Source**: `zomes/materials/coordinator/src/lib.rs`

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `create_material` | `CreateMaterialInput { name: String, material_type: MaterialType, properties: MaterialProperties, certifications: Vec<Certification>, safety_data_sheet?: String }` | `Record` | Create a new material specification. MaterialProperties includes print temps, density, tensile strength, food_safe, uv/water resistant, recyclable flags. |
| `get_material` | `ActionHash` | `Option<Record>` | Retrieve a material by its action hash. |
| `update_material` | `UpdateMaterialInput { original_action_hash: ActionHash, name?: String, material_type?: MaterialType, properties?: MaterialProperties, certifications?: Vec<Certification>, safety_data_sheet?: String }` | `Record` | Update an existing material. All fields optional except `original_action_hash`. |
| `delete_material` | `ActionHash` | `ActionHash` | Delete a material. Returns deletion action hash. |
| `get_materials_by_type` | `GetMaterialsByTypeInput { material_type: MaterialType, pagination? }` | `PaginatedResponse<Record>` | Get all materials of a specific type (PLA, PETG, ABS, ASA, TPU, Nylon, PC, PEEK, PVA, HIPS, resins, powders, Custom). (paginated) |
| `get_food_safe_materials` | `GetFoodSafeMaterialsInput { pagination? }` | `PaginatedResponse<Record>` | Get all materials where `properties.food_safe == true`. (paginated) |

---

## Verification Zome

**Source**: `zomes/verification/coordinator/src/lib.rs`

Bridges to the Knowledge hApp for epistemic scoring of safety claims. Epistemic cache TTL: 5 minutes.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `submit_verification` | `SubmitVerificationInput { design_hash: ActionHash, verification_type: VerificationType, result: VerificationResult, evidence: Vec<ActionHash>, credentials: Vec<String> }` | `Record` | Submit a verification (StructuralAnalysis, MaterialCompatibility, PrintabilityTest, SafetyReview, FoodSafeCertification, MedicalCertification, CommunityReview). Links to design and verifier. |
| `get_design_verifications` | `HashPaginationInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all verifications for a design. (paginated) |
| `get_verification_summary` | `ActionHash` (design_hash) | `VerificationSummary { design_hash, total_verifications: u32, passed: u32, failed: u32, claims_count: u32, average_confidence: f32 }` | Get aggregated verification summary. ConditionalPass counts as passed with 0.8x confidence weight. |
| `submit_safety_claim` | `SubmitClaimInput { design_hash: ActionHash, claim_type: SafetyClaimType, claim_text: String, supporting_evidence: Vec<String> }` | `Record` | Submit a safety claim with epistemic scoring from Knowledge hApp. SafetyClaimType variants: LoadCapacity, MaterialSafety, DimensionalAccuracy, TemperatureRange, ChemicalResistance, Custom. |
| `get_design_claims` | `HashPaginationInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all safety claims for a design. (paginated) |
| `get_epistemic_score` | `ActionHash` (design_hash) | `EpistemicScore { empirical: f32, normative: f32, mythic: f32, overall_confidence: f32 }` | Get aggregated epistemic score across all claims. Defaults: empirical=0.5, normative=0.3, mythic=0.2 when Knowledge hApp is unreachable. |

---

## Bridge Zome

**Source**: `zomes/bridge/coordinator/src/lib.rs`

Implements the Anticipatory Repair Loop and cross-hApp integration (Property, Knowledge, Supply Chain, HEARTH, Marketplace). All state-changing operations are rate-limited (default: 100 ops / 60s per agent, configurable via DNA properties). Operations marked with (consciousness-gated) require identity hApp consciousness verification (graceful fallback if unreachable).

### Anticipatory Repair System

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `create_repair_prediction` | `CreateRepairPredictionInput { property_asset_hash: ActionHash, asset_model: String, predicted_failure_component: String, failure_probability: f32, estimated_failure_date: Timestamp, confidence_interval_days: u32, sensor_data_summary: String }` | `Record` | Create a repair prediction from digital twin data. Validates failure_probability in [0.0, 1.0]. Auto-creates workflow if probability > 0.7. Notifies Property hApp. (consciousness-gated, rate-limited) |
| `create_repair_workflow` | `ActionHash` (prediction_hash) | `Record` | Create a repair workflow from a prediction. Initial status: Predicted. |
| `update_repair_workflow` | `UpdateWorkflowInput { workflow_hash: ActionHash, status: RepairWorkflowStatus, design_hash?, printer_hash?, hearth_funding_hash?, print_job_hash? }` | `Record` | Update repair workflow status and linked resources. Creator-only. Auto-sets completed_at for Installed/Cancelled status. (consciousness-gated) |
| `get_active_workflows` | `GetActiveWorkflowsInput { pagination? }` | `PaginatedResponse<Record>` | Get active (non-completed) repair workflows. (paginated) |

### Events

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `emit_fabrication_event` | `EmitEventInput { event_type: FabEventType, design_id?: ActionHash, payload: String }` | `Record` | Emit a fabrication event to the event log. (rate-limited) |
| `get_recent_events` | `GetRecentEventsInput { since?: Timestamp, pagination? }` | `PaginatedResponse<Record>` | Get recent fabrication events, optionally filtered by timestamp. (paginated) |

### Cross-hApp Integration

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `list_design_on_marketplace` | `ListDesignInput { design_hash: ActionHash, price?: u64, listing_type: ListingType }` | `Record` | List a design on the marketplace. Validates design exists on DHT. Notifies commons bridge. (consciousness-gated, rate-limited) |
| `link_material_to_supplier` | `LinkSupplierInput { material_hash: ActionHash, supplier_did: String, supplychain_item_hash?: ActionHash }` | `Record` | Link a material to a supply chain supplier. Validates material exists on DHT. (rate-limited) |

### Audit Trail

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_audit_trail` | `AuditTrailFilter { domain?: FabricationDomain, agent?: AgentPubKey, after?: Timestamp, before?: Timestamp, limit?: u32, pagination? }` | `PaginatedResponse<Record>` | Query the audit trail with optional filters by domain, agent, and time range. All state-changing bridge operations are automatically logged. (paginated) |

---

## Symthaea Zome

**Source**: `zomes/symthaea/coordinator/src/lib.rs`

HDC (Hyperdimensional Computing) operations and AI-assisted design generation. Encodes natural language into 4096-dimensional continuous hypervectors via FabHV/FabTextEncoder. Supports lateral binding (element-wise multiply), cosine similarity search, and LSH-accelerated approximate nearest neighbor.

### HDC Intent Creation

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `generate_intent_vector` | `CreateIntentInput { description: String, language?: String }` | `IntentResult { record: Record, bindings: Vec<SerializedBinding>, vector_hash: String }` | Encode a natural language description into a 4096D HDC hypervector. Parses semantic bindings (Base, Dimensional, Material, Modifier, Functional roles). Links to author, global anchor, and category anchor. |
| `lateral_bind` | `LateralBindInput { base_intent_hash: ActionHash, modifier_descriptions: Vec<String> }` | `IntentResult` | Combine a base intent with modifiers via HDC bind operation (element-wise multiply). E.g., bracket (x) 12mm (x) weatherproof. Creates a new composite intent entry. |

### Semantic Search

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `semantic_search` | `SemanticSearchInput { intent_hash: ActionHash, threshold?: f32, limit?: u32 }` | `Vec<SearchResult { design_hash, similarity_score: f32, matched_bindings }>` | Find designs by cosine similarity in 4096D HDC space. Brute-force scan of all intents. Default threshold from DNA config. |
| `semantic_search_by_category` | `SemanticSearchInput { intent_hash: ActionHash, threshold?: f32, limit?: u32 }` | `Vec<SearchResult>` | Category-partitioned semantic search -- scans only intents in the same category as the query for faster results. |
| `semantic_search_lsh` | `LshSearchInput { query_description: String, threshold?: f32, limit?: u32 }` | `Vec<SearchResult>` | LSH-accelerated semantic search. Builds in-memory LSH index, uses multi-probe approximate nearest neighbor. Falls back to brute-force when index < 50 intents. Accepts raw text (no pre-existing intent required). |

### Parametric Design Generation

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `generate_parametric_variant` | `GenerateVariantInput { base_design_hash: ActionHash, intent_modifiers: Vec<SerializedBinding>, material_constraints: Vec<String>, printer_constraints?: String }` | `Record` | Generate a parametric variant from intent modifiers + constraints. Builds CSG tree (symthaea-fabrication-kernel format) from shape/dimension/feature bindings. Confidence score based on recognized modifier ratio. |
| `optimize_for_local` | `OptimizeLocalInput { design_hash: ActionHash, local_materials: Vec<ActionHash>, local_printers: Vec<ActionHash>, energy_preference: String }` | `Record` | Optimize a design for local conditions (available materials, printers, energy source). Supports "solar"/"renewable"/"grid" energy preferences. |

### Repair Prediction

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `predict_repair_needs` | `PredictRepairInput { property_asset_hash: ActionHash, sensor_history: Vec<SensorReading { timestamp, sensor_type, value, unit }>, usage_hours: u32, asset_type?: String, mtbf_hours?: u32 }` | `RepairPredictionResult { predicted_component, failure_probability: f32, estimated_remaining_hours: u32, recommended_action, sensor_summary: SensorSummary, matching_repair_designs }` | Predict repair needs from digital twin sensor data. Uses Weibull-inspired failure probability with per-asset MTBF lookup. Analyzes vibration RMS/trend, temperature, torque variance, current draw, pressure drop. Sensor types: vibration, temperature, torque, current, pressure, stress_cycle. |

### Queries

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_my_intents` | `MyIntentsInput { pagination? }` | `PaginatedResponse<Record>` | Get all HDC intents created by the calling agent. (paginated) |
| `get_design_optimizations` | `HashPaginationInput { hash: ActionHash, pagination? }` | `PaginatedResponse<Record>` | Get all optimization results for a design. (paginated) |

### Consciousness-Gated Fabrication

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `generate_consciousness_gated_variant` | `ConsciousnessGatedInput { base_design_hash: ActionHash, intent_modifiers: Vec<SerializedBinding>, material_constraints: Vec<String>, printer_constraints?: String, consciousness_score: f32 }` | `Record` | Generate a parametric variant with consciousness-gated exploration depth. 5 tiers: Reflexive (<0.2, strict templates), Adaptive (<0.4), Intentional (<0.6), Creative (<0.8, novel CSG), Transcendent (>=0.8, cross-domain). Controls CSG depth, exploration width, search threshold, and min confidence. |
| `get_consciousness_params` | `f32` (consciousness_score) | `ConsciousnessGatedParams { tier, consciousness_score, max_csg_depth, exploration_width, search_threshold, min_confidence, creative_features_enabled }` | Query consciousness-gated fabrication parameters for a given score (read-only, for UI display). |

---

## Common Types

These types are used across multiple zomes (defined in `fabrication_common`):

| Type | Description |
|------|-------------|
| `PaginationInput { offset: u32, limit: u32 }` | Pagination request. Limit clamped to [1, 100]. |
| `PaginatedResponse<T> { items: Vec<T>, total: u32, offset: u32, limit: u32 }` | Paginated response wrapper. |
| `FabricationConfig` | DNA properties: rate limits, PoGF weights, HDC dimensions, similarity threshold. |
| `SafetyClass` | Class0Decorative, Class1Functional, Class2LoadBearing, Class3BodyContact, Class4Medical, Class5Critical. Class3+ requires verification before printing. |
| `DesignCategory` | Design classification enum. |
| `MaterialType` | PLA, PETG, ABS, ASA, TPU, Nylon, PC, PEEK, PVA, HIPS, StandardResin, ToughResin, FlexibleResin, CastableResin, DentalResin, NylonPowder, MetalPowder, Custom(String). |
| `License` | OpenHardware, CreativeCommons(CCVariant), Proprietary, Custom(String). ND variants block forking. |
| `TypedFabricationSignal` | All zomes emit real-time signals with `{ domain, event_type, payload }`. |
| `FabricationError` | Typed errors: NotFound, Unauthorized, ValidationFailed, RateLimited. |

---

## Security Features

- **Rate limiting**: All state-changing bridge operations enforce per-agent rate limits (configurable via DNA properties, default 100 ops / 60s)
- **Consciousness gating**: Bridge operations (`create_repair_prediction`, `update_repair_workflow`, `list_design_on_marketplace`) require identity hApp consciousness verification with graceful fallback
- **Safety class enforcement**: Class3+ designs require at least one passing verification before print jobs can be created
- **Author/owner checks**: Update and delete operations verify the caller is the original author/owner
- **Float validation**: All f32 inputs are validated with `is_finite()` guards
- **Audit trail**: All bridge state changes are automatically logged with agent, domain, timestamp, and payload

---

*Generated from 7 coordinator zomes in the Mycelix Fabrication hApp.*
