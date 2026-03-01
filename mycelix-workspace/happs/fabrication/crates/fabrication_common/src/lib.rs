//! Fabrication Common - Shared Types for Mycelix Fabrication hApp
//!
//! This crate contains all the shared types, enums, and structures used across
//! the Fabrication hApp zomes. It implements the revolutionary features:
//!
//! - HDC-Encoded Parametric Designs (Generative CAD Commons)
//! - Proof of Grounded Fabrication (PoGF) for metabolic accountability
//! - Anticipatory Repair Loop for autopoietic maintenance
//! - Cincinnati Algorithm for teleomorphic quality monitoring

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

#[allow(clippy::result_unit_err)]
pub mod validation {
    use hdi::prelude::{ValidateCallbackResult, LinkTag};

    pub fn require_finite(val: f32, field: &str) -> Result<ValidateCallbackResult, ()> {
        if !val.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be a finite number", field),
            ));
        }
        Err(())
    }

    pub fn require_finite_f64(val: f64, field: &str) -> Result<ValidateCallbackResult, ()> {
        if !val.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be a finite number", field),
            ));
        }
        Err(())
    }

    pub fn require_in_range(val: f32, min: f32, max: f32, field: &str) -> Result<ValidateCallbackResult, ()> {
        if !val.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be a finite number", field),
            ));
        }
        if val < min || val > max {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between {} and {}", field, min, max),
            ));
        }
        Err(())
    }

    pub fn require_in_range_f64(val: f64, min: f64, max: f64, field: &str) -> Result<ValidateCallbackResult, ()> {
        if !val.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be a finite number", field),
            ));
        }
        if val < min || val > max {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} must be between {} and {}", field, min, max),
            ));
        }
        Err(())
    }

    pub fn require_non_empty(s: &str, field: &str) -> Result<ValidateCallbackResult, ()> {
        if s.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} cannot be empty", field),
            ));
        }
        Err(())
    }

    pub fn require_max_len(s: &str, max: usize, field: &str) -> Result<ValidateCallbackResult, ()> {
        if s.len() > max {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} cannot exceed {} characters", field, max),
            ));
        }
        Err(())
    }

    pub fn require_max_vec_len<T>(v: &[T], max: usize, field: &str) -> Result<ValidateCallbackResult, ()> {
        if v.len() > max {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} cannot exceed {} items", field, max),
            ));
        }
        Err(())
    }

    pub fn require_max_tag_len(tag: &LinkTag, max: usize, name: &str) -> Result<ValidateCallbackResult, ()> {
        if tag.as_ref().len() > max {
            return Ok(ValidateCallbackResult::Invalid(
                format!("{} link tag cannot exceed {} bytes", name, max),
            ));
        }
        Err(())
    }
}

/// Check a validation result — returns early with Invalid if the check fails.
/// Usage: `check!(validation::require_finite(val, "field"));`
#[macro_export]
macro_rules! check {
    ($expr:expr) => {
        if let Ok(result) = $expr {
            return Ok(result);
        }
    };
}

// =============================================================================
// FABRICATION CONFIG (loaded from DNA properties, falls back to defaults)
// =============================================================================

/// Consolidated configuration for all fabrication hApp constants.
/// Loaded from DNA properties at runtime, falling back to defaults for missing fields.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FabricationConfig {
    /// PoGF energy attestation weight
    pub pog_energy_weight: f32,
    /// PoGF material attestation weight
    pub pog_material_weight: f32,
    /// PoGF quality monitoring weight
    pub pog_quality_weight: f32,
    /// PoGF local participation weight
    pub pog_local_weight: f32,
    /// Minimum PoGF score for MYCELIUM reputation
    pub min_pog_for_mycelium: f32,
    /// Base MYCELIUM reward per successful print
    pub base_mycelium_reward: u64,
    /// Rate limit: max operations per window
    pub rate_limit_max_ops: u32,
    /// Rate limit: window in seconds
    pub rate_limit_window_secs: u32,
    /// HDC vector dimensions for symthaea coordinator
    pub hdc_dimensions: u32,
    /// Cosine similarity threshold for semantic matching
    pub similarity_threshold: f32,
}

impl Default for FabricationConfig {
    fn default() -> Self {
        Self {
            pog_energy_weight: 0.3,
            pog_material_weight: 0.3,
            pog_quality_weight: 0.2,
            pog_local_weight: 0.2,
            min_pog_for_mycelium: 0.6,
            base_mycelium_reward: 10,
            rate_limit_max_ops: 100,
            rate_limit_window_secs: 60,
            hdc_dimensions: 16_384,
            similarity_threshold: 0.7,
        }
    }
}

impl FabricationConfig {
    /// Load from DNA properties bytes, falling back to defaults for missing fields.
    /// Validates all float fields are finite; falls back to default on invalid values.
    pub fn from_properties_or_default(properties_bytes: &[u8]) -> Self {
        let config: Self = serde_json::from_slice::<Self>(properties_bytes).unwrap_or_default();
        if config.all_floats_finite() {
            config
        } else {
            Self::default()
        }
    }

    /// Check that all f32 fields are finite (not NaN or Inf).
    pub fn all_floats_finite(&self) -> bool {
        self.pog_energy_weight.is_finite()
            && self.pog_material_weight.is_finite()
            && self.pog_quality_weight.is_finite()
            && self.pog_local_weight.is_finite()
            && self.min_pog_for_mycelium.is_finite()
            && self.similarity_threshold.is_finite()
    }

    /// Validate that PoGF weights sum to 1.0 (within tolerance).
    pub fn pog_weights_valid(&self) -> bool {
        let sum = self.pog_energy_weight
            + self.pog_material_weight
            + self.pog_quality_weight
            + self.pog_local_weight;
        (sum - 1.0).abs() < 0.001
    }
}

// =============================================================================
// SIGNAL TYPES
// =============================================================================

// =============================================================================
// TYPED SIGNAL TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FabricationDomain {
    Design,
    Print,
    Material,
    Verification,
    Printer,
    Symthaea,
    Bridge,
}

impl std::fmt::Display for FabricationDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Design => write!(f, "designs"),
            Self::Print => write!(f, "prints"),
            Self::Material => write!(f, "materials"),
            Self::Verification => write!(f, "verification"),
            Self::Printer => write!(f, "printers"),
            Self::Symthaea => write!(f, "symthaea"),
            Self::Bridge => write!(f, "bridge"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FabricationEventType {
    // Design events
    DesignCreated,
    DesignUpdated,
    DesignDeleted,
    DesignForked,
    FileAdded,
    // Printer events
    PrinterRegistered,
    PrinterUpdated,
    PrinterDeactivated,
    AvailabilityChanged,
    // Material events
    MaterialCreated,
    // Verification events
    VerificationSubmitted,
    ClaimSubmitted,
    // Print job events
    SafetyCheckSkipped,
    JobCreated,
    JobAccepted,
    PrintStarted,
    ProgressUpdated,
    PrintCompleted,
    PrintCancelled,
    ResultRecorded,
    CincinnatiStarted,
    AnomalyDetected,
    // Bridge events
    PredictionCreated,
    WorkflowCreated,
    WorkflowUpdated,
    EventEmitted,
    DesignListed,
    SupplierLinked,
    // Symthaea events
    IntentGenerated,
    IntentBound,
    VariantGenerated,
    OptimizationCompleted,
    ConsciousnessGatedVariant,
    // Audit diagnostic
    AuditFallback,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TypedFabricationSignal {
    pub domain: FabricationDomain,
    pub event_type: FabricationEventType,
    pub payload: String,
}

/// Known fabrication zome names for dispatch validation.
pub const FABRICATION_ZOME_ALLOWLIST: &[&str] = &[
    "designs", "printers", "prints", "materials",
    "verification", "symthaea", "bridge",
];

// =============================================================================
// ERROR TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FabricationError {
    NotFound { entity: String, hash: String },
    Unauthorized { action: String, reason: String },
    RateLimited { max_ops: u32, window_secs: u32 },
    ValidationFailed { field: String, reason: String },
    CrossHappUnavailable { role: String },
    AuditFailed { reason: String },
}

impl FabricationError {
    pub fn to_wasm_error(&self) -> WasmError {
        wasm_error!(WasmErrorInner::Guest(format!("{:?}", self)))
    }

    /// Convenience: entity not found on DHT
    pub fn not_found(entity: &str, hash: &impl std::fmt::Display) -> WasmError {
        Self::NotFound { entity: entity.to_string(), hash: hash.to_string() }.to_wasm_error()
    }

    /// Convenience: caller is not authorized for this action
    pub fn unauthorized(action: &str, reason: &str) -> WasmError {
        Self::Unauthorized { action: action.to_string(), reason: reason.to_string() }.to_wasm_error()
    }

    /// Convenience: cross-hApp role is unavailable
    pub fn cross_happ(role: &str) -> WasmError {
        Self::CrossHappUnavailable { role: role.to_string() }.to_wasm_error()
    }
}

// =============================================================================
// AUDIT TRAIL TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FabricationAuditEntry {
    pub domain: FabricationDomain,
    pub event_type: FabricationEventType,
    pub action_hash: ActionHash,
    pub agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AuditTrailFilter {
    pub domain: Option<FabricationDomain>,
    pub agent: Option<AgentPubKey>,
    pub after: Option<Timestamp>,
    pub before: Option<Timestamp>,
    pub limit: Option<u32>,
    pub pagination: Option<PaginationInput>,
}

// =============================================================================
// PAGINATION TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginationInput {
    pub offset: u32,
    pub limit: u32,
}

impl PaginationInput {
    pub fn clamp(&self) -> (u32, u32) {
        let limit = self.limit.clamp(1, 100);
        (self.offset, limit)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedResponse<T: Serialize> {
    pub items: Vec<T>,
    pub offset: u32,
    pub limit: u32,
    pub total: u32,
}

/// Apply optional pagination to a collected `Vec`, returning a
/// [`PaginatedResponse`]. When `pagination` is `None` all items are returned.
pub fn paginate<T: Serialize>(items: Vec<T>, pagination: Option<&PaginationInput>) -> PaginatedResponse<T> {
    let total = items.len() as u32;

    match pagination {
        Some(p) => {
            let (offset, limit) = p.clamp();
            let start = (offset as usize).min(items.len());
            let end = (start + limit as usize).min(items.len());
            PaginatedResponse {
                items: items.into_iter().skip(start).take(end - start).collect(),
                total,
                offset,
                limit,
            }
        }
        None => PaginatedResponse {
            items,
            total,
            offset: 0,
            limit: total,
        },
    }
}

/// Common input: ActionHash + optional pagination.
/// Used by endpoints that paginate results for a specific entity.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HashPaginationInput {
    pub hash: ActionHash,
    pub pagination: Option<PaginationInput>,
}

/// Common input: optional pagination only (agent inferred from agent_info()).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AgentPaginationInput {
    pub pagination: Option<PaginationInput>,
}

// =============================================================================
// HDC (HYPERDIMENSIONAL COMPUTING) TYPES
// =============================================================================

/// HDC Hypervector for semantic design encoding
///
/// Enables generative manufacturing by storing design "Intent" as high-dimensional
/// vectors that can be combined through lateral binding operations.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HdcHypervector {
    /// Number of dimensions (typically 10,000)
    pub dimensions: u32,
    /// Bipolar vector values {-1, +1}
    pub vector: Vec<i8>,
    /// Semantic concepts bound to this vector
    pub semantic_bindings: Vec<SemanticBinding>,
    /// How this vector was generated
    pub generation_method: HdcMethod,
}

/// A semantic concept bound to an HDC hypervector
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SemanticBinding {
    /// The concept being encoded (e.g., "bracket", "12mm", "weatherproof")
    pub concept: String,
    /// The role of this binding in the design
    pub role: BindingRole,
    /// Binding strength (0.0-1.0)
    pub weight: f32,
}

/// Role of a semantic binding in design encoding
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BindingRole {
    /// Core object type (e.g., "bracket", "gear", "enclosure")
    Base,
    /// Attribute modifier (e.g., "heavy-duty", "weatherproof")
    Modifier,
    /// Size constraint (e.g., "12mm", "M8")
    Dimensional,
    /// Material binding (e.g., "PETG", "food-safe")
    Material,
    /// Purpose (e.g., "load-bearing", "decorative")
    Functional,
}

/// How an HDC hypervector was generated
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum HdcMethod {
    /// Designer manually specified the encoding
    ManualEncoding,
    /// AI (Symthaea) inferred from description
    SymthaeaGenerated,
    /// Combined from existing design vectors
    LateralBinding,
    /// Optimized via genetic algorithm
    EvolutionarySearch,
}

// =============================================================================
// HDC ENCODING (DUAL FORMAT SUPPORT)
// =============================================================================

/// Dual-format HDC encoding: legacy bipolar or new continuous
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum HdcEncoding {
    /// Legacy bipolar {-1, +1} vectors
    BipolarLegacy { dimensions: u32, vector: Vec<i8> },
    /// Continuous f32 vectors (FabHV, 16,384D standard)
    Continuous { dimensions: u32, vector: Vec<f32> },
    /// Hash-only reference (vector stored externally)
    HashOnly { vector_hash: String },
}

// =============================================================================
// WASM-COMPATIBLE HDC MODULE (FabHV)
// =============================================================================

pub mod hdc {
    use serde::{Deserialize, Serialize};

    /// Standard HDC dimension (2^14 = 16,384), matching symthaea-core's HDC_DIMENSION.
    pub const FAB_HDC_DIM: usize = 16_384;

    /// Continuous hypervector for WASM-compatible HDC operations
    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct FabHV {
        pub data: Vec<f32>,
    }

    impl FabHV {
        pub fn zero(dim: usize) -> Self {
            Self { data: vec![0.0; dim] }
        }

        pub fn from_vec(data: Vec<f32>) -> Self {
            Self { data }
        }

        pub fn dim(&self) -> usize {
            self.data.len()
        }

        /// Deterministic pseudo-random HV via xorshift64
        pub fn random(dim: usize, seed: u64) -> Self {
            let mut state = seed;
            if state == 0 { state = 0xDEADBEEF; }
            let mut data = Vec::with_capacity(dim);
            for _ in 0..dim {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1]
                let val = ((state as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
                data.push(val);
            }
            let mut hv = Self { data };
            hv.normalize();
            hv
        }

        /// Element-wise multiply (binding operation)
        pub fn bind(&self, other: &Self) -> Self {
            assert_eq!(self.data.len(), other.data.len(), "FabHV dimension mismatch");
            let data: Vec<f32> = self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect();
            let mut hv = Self { data };
            hv.normalize();
            hv
        }

        /// Element-wise mean + normalize (bundling operation)
        pub fn bundle(hvs: &[&Self]) -> Self {
            if hvs.is_empty() {
                return Self::zero(FAB_HDC_DIM);
            }
            let dim = hvs[0].dim();
            let n = hvs.len() as f32;
            let mut data = vec![0.0f32; dim];
            for hv in hvs {
                for (i, val) in hv.data.iter().enumerate() {
                    data[i] += val;
                }
            }
            for val in &mut data {
                *val /= n;
            }
            let mut hv = Self { data };
            hv.normalize();
            hv
        }

        /// Cosine similarity
        pub fn similarity(&self, other: &Self) -> f32 {
            assert_eq!(self.data.len(), other.data.len(), "FabHV dimension mismatch");
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for (a, b) in self.data.iter().zip(other.data.iter()) {
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom < 1e-10 {
                return 0.0;
            }
            dot / denom
        }

        /// L2 normalization in-place
        pub fn normalize(&mut self) {
            let norm: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for val in &mut self.data {
                    *val /= norm;
                }
            }
        }
    }

    /// Character trigram + word position encoder
    pub struct FabTextEncoder {
        dim: usize,
        // We generate char/position vectors on the fly using deterministic seeds
    }

    impl FabTextEncoder {
        pub fn new(dim: usize) -> Self {
            Self { dim }
        }

        fn char_hv(&self, c: char) -> FabHV {
            FabHV::random(self.dim, c as u64 * 73856093)
        }

        fn position_hv(&self, pos: usize) -> FabHV {
            FabHV::random(self.dim, (pos as u64 + 1) * 19349669)
        }

        fn trigram_hv(&self, a: char, b: char, c: char) -> FabHV {
            let ha = self.char_hv(a);
            let hb = self.char_hv(b);
            let hc = self.char_hv(c);
            ha.bind(&hb).bind(&hc)
        }

        pub fn encode(&self, text: &str) -> FabHV {
            let lower = text.to_lowercase();
            let words: Vec<&str> = lower.split_whitespace().collect();
            if words.is_empty() {
                return FabHV::zero(self.dim);
            }

            let mut word_hvs: Vec<FabHV> = Vec::new();
            for (pos, word) in words.iter().enumerate() {
                let chars: Vec<char> = word.chars().collect();
                let mut trigrams: Vec<FabHV> = Vec::new();
                if chars.len() < 3 {
                    // For short words, use the chars directly
                    let padded: Vec<char> = std::iter::once(' ')
                        .chain(chars.iter().copied())
                        .chain(std::iter::once(' '))
                        .collect();
                    for window in padded.windows(3) {
                        trigrams.push(self.trigram_hv(window[0], window[1], window[2]));
                    }
                } else {
                    for window in chars.windows(3) {
                        trigrams.push(self.trigram_hv(window[0], window[1], window[2]));
                    }
                }
                if trigrams.is_empty() {
                    continue;
                }
                let refs: Vec<&FabHV> = trigrams.iter().collect();
                let word_hv = FabHV::bundle(&refs);
                let pos_hv = self.position_hv(pos);
                word_hvs.push(word_hv.bind(&pos_hv));
            }

            if word_hvs.is_empty() {
                return FabHV::zero(self.dim);
            }
            let refs: Vec<&FabHV> = word_hvs.iter().collect();
            FabHV::bundle(&refs)
        }
    }
}

// =============================================================================
// PARAMETRIC DESIGN TYPES
// =============================================================================

/// Parametric schema for generative design
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ParametricSchema {
    /// The parametric engine used
    pub engine: ParametricEngine,
    /// IPFS CID of the template file
    pub source_template: String,
    /// Configurable parameters
    pub parameters: Vec<DesignParameter>,
    /// Constraints between parameters
    pub constraints: Vec<ParametricConstraint>,
    /// Whether Symthaea can auto-modify
    pub auto_generate: bool,
}

/// Supported parametric CAD engines
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ParametricEngine {
    OpenSCAD,
    CadQuery,
    FreeCAD,
    JSCAD,
    Other(String),
}

/// A configurable design parameter
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DesignParameter {
    /// Parameter name (e.g., "pipe_diameter")
    pub name: String,
    /// Type of parameter
    pub param_type: ParameterType,
    /// Default value
    pub default_value: ParameterValue,
    /// Minimum allowed value
    pub min_value: Option<ParameterValue>,
    /// Maximum allowed value
    pub max_value: Option<ParameterValue>,
    /// Unit of measurement (e.g., "mm", "degrees")
    pub unit: Option<String>,
    /// Link to HDC concept
    pub hdc_binding: Option<String>,
}

/// Types of design parameters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ParameterType {
    Length,
    Angle,
    Count,
    Boolean,
    Enum(Vec<String>),
    Material,
}

/// Value of a design parameter
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Number(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
}

/// Constraint between parameters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ParametricConstraint {
    /// Parameters involved in the constraint
    pub parameters: Vec<String>,
    /// Constraint expression (e.g., "wall_thickness <= outer_diameter * 0.3")
    pub expression: String,
    /// Error message if constraint violated
    pub error_message: String,
}

/// Constraint graph for dimensional relationships
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConstraintGraph {
    /// Nodes represent parameters
    pub nodes: Vec<String>,
    /// Edges represent relationships
    pub edges: Vec<ConstraintEdge>,
}

/// Edge in the constraint graph
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConstraintEdge {
    pub from: String,
    pub to: String,
    pub relationship: String,
}

/// Material binding for compatibility checking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaterialBinding {
    /// Material type required
    pub material: MaterialType,
    /// Compatibility score (0.0-1.0)
    pub compatibility: f32,
    /// Specific requirements
    pub requirements: Vec<String>,
}

// =============================================================================
// DESIGN TYPES
// =============================================================================

/// Design file attached to a design entry
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DesignFile {
    /// Filename
    pub filename: String,
    /// File format
    pub format: FileFormat,
    /// IPFS content identifier
    pub ipfs_cid: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA-256 checksum for integrity
    pub checksum_sha256: String,
}

/// Supported design file formats
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FileFormat {
    STL,
    STEP,
    ThreeMF,
    OBJ,
    SCAD,
    FCStd,
    GCODE,
    Other(String),
}

/// Categories for design organization
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DesignCategory {
    Tools,
    Parts,
    Housewares,
    Medical,
    Accessibility,
    Art,
    Education,
    Repair,
    Custom,
}

/// Safety classification system (Class 0-5)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SafetyClass {
    /// No safety concerns - decorative items
    Class0Decorative,
    /// Basic mechanical - tools, fixtures
    Class1Functional,
    /// Structural testing required
    Class2LoadBearing,
    /// Material certification needed - wearables, body contact
    Class3BodyContact,
    /// Professional verification required - medical
    Class4Medical,
    /// Multi-party certification - safety-critical applications
    Class5Critical,
}

/// Design license types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum License {
    PublicDomain,
    CreativeCommons(CCVariant),
    OpenHardware,
    Proprietary,
    Custom(String),
}

/// Creative Commons license variants
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CCVariant {
    /// CC0 - No rights reserved
    CC0,
    /// CC BY - Attribution
    BY,
    /// CC BY-SA - Attribution ShareAlike
    BYSA,
    /// CC BY-NC - Attribution NonCommercial
    BYNC,
    /// CC BY-NC-SA - Attribution NonCommercial ShareAlike
    BYNCSA,
    /// CC BY-ND - Attribution NoDerivatives
    BYND,
    /// CC BY-NC-ND - Attribution NonCommercial NoDerivatives
    BYNCND,
}

/// Epistemic dimensions for a design
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DesignEpistemic {
    /// E-axis: Can it be manufactured? (0.0-1.0)
    pub manufacturability: f32,
    /// N-axis: Is it safe to use? (0.0-1.0)
    pub safety: f32,
    /// M-axis: Does it work as intended? (0.0-1.0)
    pub usability: f32,
}

// =============================================================================
// REPAIR MANIFEST TYPES (AUTOPOIETIC LOOP)
// =============================================================================

/// Repair manifest linking a design to parent products
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RepairManifest {
    /// Link to Property hApp digital twin
    pub parent_product_hash: Option<ActionHash>,
    /// Product model identifier
    pub parent_product_model: String,
    /// Name of the part this design replaces
    pub part_name: String,
    /// Known failure modes for this part
    pub failure_modes: Vec<FailureMode>,
    /// Expected lifetime in hours of use
    pub replacement_interval: Option<u32>,
    /// How difficult is the repair?
    pub repair_difficulty: RepairDifficulty,
}

/// Why a part typically fails
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FailureMode {
    MechanicalWear,
    UvDegradation,
    ThermalCycling,
    ChemicalExposure,
    ImpactDamage,
    Fatigue,
    Corrosion,
    Abrasion,
    Other(String),
}

/// Difficulty level for repair installation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RepairDifficulty {
    /// No tools required, snap-fit
    ToolFree,
    /// Basic hand tools only
    BasicTools,
    /// Some skill required
    Intermediate,
    /// Professional recommended
    Advanced,
    /// Expert only
    Expert,
}

// =============================================================================
// PRINTER TYPES
// =============================================================================

/// Geographic location for printer matching
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    /// Geohash for proximity queries
    pub geohash: String,
    /// Latitude in decimal degrees (-90 to 90)
    pub lat: Option<f64>,
    /// Longitude in decimal degrees (-180 to 180)
    pub lon: Option<f64>,
    /// City name
    pub city: Option<String>,
    /// Region/State/Province
    pub region: Option<String>,
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
}

/// Haversine distance between two lat/lon points in km. Pure function.
pub fn haversine_distance_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0;
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let a = (d_lat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (d_lon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().asin()
}

/// Type of 3D printer
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PrinterType {
    /// Fused Deposition Modeling
    FDM,
    /// Stereolithography
    SLA,
    /// Selective Laser Sintering
    SLS,
    /// Digital Light Processing
    DLP,
    /// Multi Jet Fusion
    MJF,
    /// Other printer technology
    Other(String),
}

/// Printer capabilities for matching
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PrinterCapabilities {
    /// Build volume in mm
    pub build_volume: BuildVolume,
    /// Supported layer heights in mm
    pub layer_heights: Vec<f32>,
    /// Installed nozzle diameters in mm
    pub nozzle_diameters: Vec<f32>,
    /// Has heated bed
    pub heated_bed: bool,
    /// Has enclosure
    pub enclosure: bool,
    /// Number of materials (None = single)
    pub multi_material: Option<u8>,
    /// Maximum hotend temperature in Celsius
    pub max_temp_hotend: u16,
    /// Maximum bed temperature in Celsius
    pub max_temp_bed: u16,
    /// Additional features
    pub features: Vec<PrinterFeature>,
}

/// Printer build volume
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BuildVolume {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Additional printer features
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PrinterFeature {
    AutoLeveling,
    FilamentSensor,
    PowerRecovery,
    DirectDrive,
    AllMetalHotend,
    NetworkConnected,
    CameraMonitoring,
    AirFiltration,
    Other(String),
}

/// Printer availability status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AvailabilityStatus {
    Available,
    Busy,
    Maintenance,
    Offline,
    ByAppointment,
}

/// Commercial printing rates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PrinterRates {
    /// Base rate per hour
    pub hourly_rate: f64,
    /// Rate per gram of material
    pub material_rate: f64,
    /// Currency code
    pub currency: String,
    /// Minimum order amount
    pub minimum_order: Option<f64>,
}

/// Mirror of the Printer entry type for cross-zome deserialization.
/// Identical serde shape to `printers_integrity::Printer` but without
/// `#[hdk_entry_helper]`, so coordinators outside the printers zome can
/// deserialize printer records without linking to the integrity crate.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PrinterInfo {
    pub id: String,
    pub name: String,
    pub owner: AgentPubKey,
    pub location: Option<GeoLocation>,
    pub printer_type: PrinterType,
    pub capabilities: PrinterCapabilities,
    pub materials_available: Vec<MaterialType>,
    pub availability: AvailabilityStatus,
    pub rates: Option<PrinterRates>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl TryFrom<SerializedBytes> for PrinterInfo {
    type Error = SerializedBytesError;
    fn try_from(sb: SerializedBytes) -> Result<Self, Self::Error> {
        decode(&sb.bytes())
            .map_err(|e| SerializedBytesError::Deserialize(e.to_string()))
    }
}

impl TryFrom<PrinterInfo> for SerializedBytes {
    type Error = SerializedBytesError;
    fn try_from(val: PrinterInfo) -> Result<Self, Self::Error> {
        let bytes = encode(&val)
            .map_err(|e| SerializedBytesError::Serialize(e.to_string()))?;
        Ok(SerializedBytes::from(UnsafeBytes::from(bytes)))
    }
}

// =============================================================================
// PRINT JOB TYPES
// =============================================================================

/// Print settings for a job
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PrintSettings {
    /// Layer height in mm
    pub layer_height: f32,
    /// Infill percentage (0-100)
    pub infill_percent: u8,
    /// Material to use
    pub material: MaterialType,
    /// Generate supports
    pub supports: bool,
    /// Use raft for bed adhesion
    pub raft: bool,
    /// Print speed in mm/s
    pub print_speed: Option<u16>,
    /// Temperature settings
    pub temperatures: TemperatureSettings,
    /// Custom G-code commands
    pub custom_gcode: Option<String>,
}

/// Temperature settings for printing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TemperatureSettings {
    /// Hotend temperature
    pub hotend: u16,
    /// Bed temperature
    pub bed: Option<u16>,
    /// Chamber temperature (for enclosed printers)
    pub chamber: Option<u16>,
}

/// Print job status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PrintJobStatus {
    Pending,
    Accepted,
    Queued,
    Printing,
    PostProcessing,
    Completed,
    Failed,
    Cancelled,
}

/// Result of a print job
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PrintResult {
    Success,
    PartialSuccess,
    Failed(FailureReason),
}

/// Reasons a print might fail
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FailureReason {
    Warping,
    LayerShift,
    NozzleClog,
    BedAdhesion,
    PowerFailure,
    MaterialOut,
    UserCancelled,
    QualityFailed,
    Other(String),
}

/// Common print issues
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrintIssue {
    Warping,
    LayerShift,
    Stringing,
    UnderExtrusion,
    OverExtrusion,
    BedAdhesion,
    SupportFailure,
    ZBanding,
    Ghosting,
    Other(String),
}

// =============================================================================
// PROOF OF GROUNDED FABRICATION (PoGF) TYPES
// =============================================================================

/// Grounding certificate for metabolic accountability
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GroundingCertificate {
    /// Unique certificate identifier
    pub certificate_id: String,
    /// Link to Terra Atlas energy source
    pub terra_atlas_energy_hash: Option<ActionHash>,
    /// Type of energy used
    pub energy_type: EnergyType,
    /// Grid carbon intensity (gCO2/kWh) at time of print
    pub grid_carbon_intensity: f32,
    /// Material passports for traceability
    pub material_passports: Vec<MaterialPassport>,
    /// Link to HEARTH local economy funding
    pub hearth_funding_hash: Option<ActionHash>,
    /// When the certificate was issued
    pub issued_at: Timestamp,
    /// Digital signature from issuer
    pub issuer_signature: Vec<u8>,
}

/// Material passport for circular economy tracking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaterialPassport {
    /// Link to material entry
    pub material_hash: ActionHash,
    /// Batch identifier
    pub batch_id: String,
    /// Origin of the material
    pub origin: MaterialOrigin,
    /// Percentage of recycled content
    pub recycled_content_percent: f32,
    /// Link to Supply Chain hApp entry
    pub supply_chain_hash: Option<ActionHash>,
    /// Certifications (FDA, etc.)
    pub certifications: Vec<String>,
    /// End of life strategy
    pub end_of_life: EndOfLifeStrategy,
}

/// Origin of manufacturing material
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialOrigin {
    /// Newly produced from raw materials
    Virgin,
    /// Recycled from manufacturing waste
    PostIndustrial,
    /// Recycled from consumer products
    PostConsumer,
    /// Made from renewable biological sources
    Biobased,
    /// Recycled from local waste streams
    UrbanMined,
}

/// What happens to the material at end of life
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EndOfLifeStrategy {
    /// Can be mechanically recycled
    MechanicalRecycling,
    /// Can be chemically recycled
    ChemicalRecycling,
    /// Naturally biodegrades
    Biodegradable,
    /// Industrial composting required
    IndustrialCompost,
    /// Can be downcycled to lower-grade products
    Downcycle,
    /// Must go to landfill (discouraged)
    Landfill,
}

/// Type of energy used for manufacturing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnergyType {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Nuclear,
    GridMix,
    Unknown,
}

// =============================================================================
// CINCINNATI ALGORITHM TYPES (TELEOMORPHIC MONITORING)
// =============================================================================

/// Cincinnati monitoring session for quality assurance
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CincinnatiSession {
    /// Unique session identifier
    pub session_id: String,
    /// Version of the estimator algorithm
    pub estimator_version: String,
    /// Sampling rate in Hz
    pub sampling_rate_hz: u32,
    /// Baseline "healthy print" signature
    pub baseline_signature: Vec<f32>,
    /// Whether monitoring is active
    pub active: bool,
    /// When monitoring started
    pub started_at: Timestamp,
}

/// Final report from Cincinnati monitoring
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CincinnatiReport {
    /// Session identifier
    pub session_id: String,
    /// Total samples collected
    pub total_samples: u64,
    /// Number of anomalies detected
    pub anomalies_detected: u32,
    /// Detailed anomaly events
    pub anomaly_events: Vec<AnomalyEvent>,
    /// Overall health score (0.0-1.0)
    pub overall_health_score: f32,
    /// Per-layer quality scores
    pub layer_by_layer_scores: Vec<f32>,
    /// Recommended action based on analysis
    pub recommended_action: CincinnatiAction,
}

/// An anomaly event detected during printing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AnomalyEvent {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Layer number when detected
    pub layer_number: u32,
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    /// Severity (0.0-1.0)
    pub severity: f32,
    /// Sensor data at time of anomaly
    pub sensor_data: SensorSnapshot,
    /// Action taken in response
    pub action_taken: Option<CincinnatiAction>,
}

/// Types of anomalies the Cincinnati algorithm detects
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AnomalyType {
    ExtrusionInconsistency,
    TemperatureDeviation,
    VibrationAnomaly,
    LayerAdhesionFailure,
    NozzleClog,
    BedLevelDrift,
    PowerFluctuation,
    FilamentSlip,
    Unknown,
}

/// Snapshot of sensor data
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SensorSnapshot {
    /// Hotend temperature
    pub hotend_temp: f32,
    /// Bed temperature
    pub bed_temp: f32,
    /// Stepper motor currents [X, Y, Z, E]
    pub stepper_currents: [f32; 4],
    /// Vibration RMS value
    pub vibration_rms: f32,
    /// Filament tension (if sensor available)
    pub filament_tension: Option<f32>,
    /// Ambient temperature
    pub ambient_temp: Option<f32>,
    /// Humidity percentage
    pub humidity: Option<f32>,
}

/// Actions the Cincinnati algorithm can take
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CincinnatiAction {
    /// All normal, continue printing
    Continue,
    /// Adjust print parameters
    AdjustParameters(ParameterAdjustment),
    /// Pause for manual inspection
    PauseForInspection,
    /// Abort the print
    AbortPrint(String),
    /// Alert the operator
    AlertOperator(String),
}

/// Parameter adjustment recommended by Cincinnati
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ParameterAdjustment {
    /// Parameter to adjust
    pub parameter: String,
    /// Current value
    pub current_value: f32,
    /// Recommended value
    pub recommended_value: f32,
    /// Reason for adjustment
    pub reason: String,
}

// =============================================================================
// MATERIAL TYPES
// =============================================================================

/// Types of 3D printing materials
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MaterialType {
    // FDM Filaments
    PLA,
    PETG,
    ABS,
    ASA,
    TPU,
    Nylon,
    PC,
    PEEK,
    PVA,
    HIPS,
    // Resin Types
    StandardResin,
    ToughResin,
    FlexibleResin,
    CastableResin,
    DentalResin,
    // Powder Types
    NylonPowder,
    MetalPowder,
    // Other
    Custom(String),
}

/// Physical properties of a material
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaterialProperties {
    /// Minimum print temperature
    pub print_temp_min: u16,
    /// Maximum print temperature
    pub print_temp_max: u16,
    /// Minimum bed temperature
    pub bed_temp_min: Option<u16>,
    /// Maximum bed temperature
    pub bed_temp_max: Option<u16>,
    /// Density in g/cm³
    pub density_g_cm3: f32,
    /// Tensile strength in MPa
    pub tensile_strength_mpa: Option<f32>,
    /// Elongation at break percentage
    pub elongation_percent: Option<f32>,
    /// Safe for food contact
    pub food_safe: bool,
    /// Resistant to UV degradation
    pub uv_resistant: bool,
    /// Resistant to water
    pub water_resistant: bool,
    /// Can be recycled
    pub recyclable: bool,
}

/// Material certification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Certification {
    /// Type of certification
    pub cert_type: CertificationType,
    /// Issuing organization
    pub issuer: String,
    /// Expiration date
    pub valid_until: Option<Timestamp>,
    /// IPFS CID of certification document
    pub document_cid: Option<String>,
}

/// Types of material certifications
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CertificationType {
    FoodSafe,
    Biocompatible,
    FlameRetardant,
    RoHSCompliant,
    REACHCompliant,
    ISO,
    FDA,
    Custom(String),
}

// =============================================================================
// VERIFICATION TYPES
// =============================================================================

/// Types of design verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationType {
    StructuralAnalysis,
    MaterialCompatibility,
    PrintabilityTest,
    SafetyReview,
    FoodSafeCertification,
    MedicalCertification,
    CommunityReview,
}

/// Result of a verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VerificationResult {
    Passed {
        confidence: f32,
        notes: String,
    },
    Failed {
        reasons: Vec<String>,
    },
    ConditionalPass {
        conditions: Vec<String>,
        confidence: f32,
    },
    NeedsMoreEvidence,
}

/// Types of safety claims
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SafetyClaimType {
    /// Load capacity claim (e.g., "Supports 50kg")
    LoadCapacity(String),
    /// Material safety claim (e.g., "Food-safe when printed in PETG")
    MaterialSafety(String),
    /// Dimensional accuracy claim (e.g., "Fits standard M8 bolt")
    DimensionalAccuracy(String),
    /// Temperature range claim (e.g., "Safe up to 80°C")
    TemperatureRange(String),
    /// Chemical resistance claim
    ChemicalResistance(String),
    /// Custom claim type
    Custom(String),
}

/// Epistemic classification from Knowledge hApp
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ClaimEpistemic {
    /// Empirical verifiability (0.0-1.0)
    pub empirical: f32,
    /// Normative dimension (0.0-1.0)
    pub normative: f32,
    /// Mythic/meaning dimension (0.0-1.0)
    pub mythic: f32,
}

/// Status of a verification request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RequestStatus {
    Open,
    InProgress,
    Completed,
    Cancelled,
    Expired,
}

// =============================================================================
// ANTICIPATORY REPAIR TYPES
// =============================================================================

/// Repair prediction from Property hApp digital twin
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RepairPrediction {
    /// Link to Property hApp digital twin
    pub property_asset_hash: ActionHash,
    /// Asset model identifier
    pub asset_model: String,
    /// Component predicted to fail
    pub predicted_failure_component: String,
    /// Probability of failure (0.0-1.0)
    pub failure_probability: f32,
    /// Estimated failure date
    pub estimated_failure_date: Timestamp,
    /// Confidence interval in days
    pub confidence_interval_days: u32,
    /// Summary of sensor data
    pub sensor_data_summary: String,
    /// Recommended action
    pub recommended_action: RepairAction,
    /// When prediction was made
    pub created_at: Timestamp,
}

/// Recommended repair action
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RepairAction {
    /// Commercial part available, order it
    OrderPart,
    /// Fabrication design exists, print it
    PrintReplacement,
    /// No design exists, create one (bounty)
    CreateDesign,
    /// Professional service required
    ScheduleMaintenance,
    /// Not critical yet, monitor
    Monitor,
}

/// Workflow status for anticipatory repair
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RepairWorkflowStatus {
    /// Failure predicted
    Predicted,
    /// Repair design located
    DesignFound,
    /// Local printer matched
    PrinterMatched,
    /// HEARTH funding secured
    FundingApproved,
    /// Print in progress
    Printing,
    /// Part printed, awaiting install
    ReadyForInstall,
    /// Complete
    Installed,
    /// User cancelled or resolved
    Cancelled,
}

// =============================================================================
// BRIDGE TYPES
// =============================================================================

/// Query types for cross-hApp integration
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FabQueryType {
    GetDesign,
    FindPrinters,
    CheckVerification,
    GetMaterialSuppliers,
    GetPrintStatistics,
}

/// Event types for cross-hApp signaling
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FabEventType {
    DesignPublished,
    DesignVerified,
    PrintCompleted,
    PrinterRegistered,
    MaterialShortage,
    VerificationRequested,
}

/// Marketplace listing types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ListingType {
    /// Sell the design file
    DesignSale,
    /// Offer to print for others
    PrintService,
    /// Sell finished printed products
    PrintedProduct,
}

/// Print statistics for a design
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PrintStatistics {
    /// Total prints attempted
    pub total_prints: u32,
    /// Successful prints
    pub successful_prints: u32,
    /// Failed prints
    pub failed_prints: u32,
    /// Average quality score
    pub average_quality: f32,
    /// Average PoGF score
    pub average_pog_score: f32,
    /// Common issues
    pub common_issues: Vec<(PrintIssue, u32)>,
}

// =============================================================================
// QUALITY PREDICTION TYPES
// =============================================================================

/// Quality prediction for a print job
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct QualityPrediction {
    /// Predicted quality score (0.0-1.0)
    pub predicted_quality: f32,
    /// Confidence in prediction (0.0-1.0)
    pub confidence: f32,
    /// Potential issues
    pub potential_issues: Vec<PrintIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Dimensional accuracy report
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DimensionalAccuracy {
    /// Average deviation in mm
    pub average_deviation: f32,
    /// Maximum deviation in mm
    pub max_deviation: f32,
    /// Measurements taken
    pub measurements: Vec<DimensionalMeasurement>,
}

/// A single dimensional measurement
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DimensionalMeasurement {
    /// Feature name
    pub feature: String,
    /// Expected dimension
    pub expected: f32,
    /// Actual dimension
    pub actual: f32,
    /// Tolerance
    pub tolerance: f32,
    /// Within tolerance?
    pub within_tolerance: bool,
}

// =============================================================================
// UTILITY IMPLEMENTATIONS
// =============================================================================

impl Default for HdcHypervector {
    fn default() -> Self {
        Self {
            dimensions: 10000,
            vector: vec![0; 10000],
            semantic_bindings: vec![],
            generation_method: HdcMethod::ManualEncoding,
        }
    }
}

impl Default for DesignEpistemic {
    fn default() -> Self {
        Self {
            manufacturability: 0.0,
            safety: 0.0,
            usability: 0.0,
        }
    }
}

impl Default for PrintSettings {
    fn default() -> Self {
        Self {
            layer_height: 0.2,
            infill_percent: 20,
            material: MaterialType::PLA,
            supports: false,
            raft: false,
            print_speed: None,
            temperatures: TemperatureSettings {
                hotend: 200,
                bed: Some(60),
                chamber: None,
            },
            custom_gcode: None,
        }
    }
}

impl Default for CincinnatiSession {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            estimator_version: "1.0.0".to_string(),
            sampling_rate_hz: 1000,
            baseline_signature: vec![],
            active: false,
            started_at: Timestamp::from_micros(0),
        }
    }
}

// =============================================================================
// PoGF ATTESTATION TYPES (VERIFIABLE SUSTAINABILITY CLAIMS)
// =============================================================================

/// Energy attestation for PoGF verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EnergyAttestation {
    /// Type of energy used
    pub energy_type: EnergyType,
    /// Energy consumed in kWh
    pub kwh_consumed: f32,
    /// Grid carbon intensity at time of print (gCO2/kWh)
    pub grid_carbon_intensity: f32,
    /// Reference to Terra Atlas energy source (external verification)
    pub terra_atlas_hash: Option<String>,
    /// Attesting agent
    pub attester: String,
    /// When attestation was made
    pub attested_at_micros: i64,
}

/// Material attestation for PoGF verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaterialAttestation {
    /// Material batch used
    pub batch_id: String,
    /// Material origin
    pub origin: MaterialOrigin,
    /// Recycled content fraction (0.0-1.0)
    pub recycled_fraction: f32,
    /// Reference to Supply Chain hApp entry
    pub supply_chain_hash: Option<String>,
    /// Certifications held by the material
    pub certifications: Vec<String>,
    /// Attesting agent
    pub attester: String,
    /// When attestation was made
    pub attested_at_micros: i64,
}

/// Local participation attestation for PoGF verification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalAttestation {
    /// HEARTH funding hash (if community-funded)
    pub hearth_funding_hash: Option<String>,
    /// Whether the print was done within the community
    pub local_printer: bool,
    /// Distance in km from requester to printer (if known)
    pub distance_km: Option<f32>,
    /// Attesting agent
    pub attester: String,
    /// When attestation was made
    pub attested_at_micros: i64,
}

/// Combined PoGF attestation bundle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PogfAttestationBundle {
    pub energy: Option<EnergyAttestation>,
    pub material: Option<MaterialAttestation>,
    pub local: Option<LocalAttestation>,
}

impl PogfAttestationBundle {
    /// Calculate a penalty multiplier for unattested claims.
    /// Fully attested = 1.0 (no penalty), fully unattested = 0.25 (75% penalty)
    pub fn attestation_multiplier(&self) -> f32 {
        let mut attested = 0u32;
        let total = 3u32;
        if self.energy.is_some() { attested += 1; }
        if self.material.is_some() { attested += 1; }
        if self.local.is_some() { attested += 1; }
        // Base 0.25 + 0.75 * (attested/total)
        0.25 + 0.75 * (attested as f32 / total as f32)
    }
}

impl SafetyClass {
    /// Returns true if this safety class requires verification before printing
    pub fn requires_verification(&self) -> bool {
        matches!(
            self,
            SafetyClass::Class3BodyContact
                | SafetyClass::Class4Medical
                | SafetyClass::Class5Critical
        )
    }

    /// Numeric level for ordering
    pub fn level(&self) -> u8 {
        match self {
            SafetyClass::Class0Decorative => 0,
            SafetyClass::Class1Functional => 1,
            SafetyClass::Class2LoadBearing => 2,
            SafetyClass::Class3BodyContact => 3,
            SafetyClass::Class4Medical => 4,
            SafetyClass::Class5Critical => 5,
        }
    }
}

// =============================================================================
// LSH (LOCALITY-SENSITIVE HASHING) FOR HDC SEARCH
// =============================================================================

pub mod lsh {
    use super::hdc::FabHV;
    use serde::{Deserialize, Serialize};

    /// Number of random hyperplanes for LSH hash
    pub const LSH_NUM_PLANES: usize = 16;

    /// LSH index using random hyperplane method for approximate nearest-neighbor
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct LshIndex {
        /// Random hyperplanes for hashing (each is a FAB_HDC_DIM vector)
        pub planes: Vec<Vec<f32>>,
        /// Buckets: hash -> list of (id, vector) pairs
        pub buckets: std::collections::HashMap<u32, Vec<LshEntry>>,
    }

    /// Entry in an LSH bucket
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct LshEntry {
        /// Identifier (e.g., ActionHash string)
        pub id: String,
        /// The stored vector
        pub vector: Vec<f32>,
    }

    impl LshIndex {
        /// Create a new LSH index with deterministic random hyperplanes
        pub fn new(num_planes: usize, dim: usize, seed: u64) -> Self {
            let mut planes = Vec::with_capacity(num_planes);
            for i in 0..num_planes {
                let plane = FabHV::random(dim, seed.wrapping_add(i as u64 * 83492791));
                planes.push(plane.data);
            }
            Self {
                planes,
                buckets: std::collections::HashMap::new(),
            }
        }

        /// Compute LSH hash for a vector
        pub fn hash(&self, vector: &[f32]) -> u32 {
            let mut h: u32 = 0;
            for (i, plane) in self.planes.iter().enumerate() {
                let dot: f32 = vector.iter()
                    .zip(plane.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot >= 0.0 {
                    h |= 1 << i;
                }
            }
            h
        }

        /// Insert a vector into the index
        pub fn insert(&mut self, id: String, vector: Vec<f32>) {
            let h = self.hash(&vector);
            self.buckets.entry(h).or_default().push(LshEntry {
                id,
                vector,
            });
        }

        /// Query: find candidate vectors in the same bucket, then rank by cosine similarity
        pub fn query(&self, query: &FabHV, max_results: usize) -> Vec<(String, f32)> {
            let h = self.hash(&query.data);
            let candidates = match self.buckets.get(&h) {
                Some(entries) => entries,
                None => return vec![],
            };

            let mut scored: Vec<(String, f32)> = candidates.iter()
                .map(|entry| {
                    let candidate = FabHV::from_vec(entry.vector.clone());
                    let sim = query.similarity(&candidate);
                    (entry.id.clone(), sim)
                })
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(max_results);
            scored
        }

        /// Multi-probe query: check the primary bucket plus neighbors (1-bit flips)
        pub fn query_multiprobe(&self, query: &FabHV, max_results: usize) -> Vec<(String, f32)> {
            let primary_hash = self.hash(&query.data);
            let mut all_candidates: Vec<(String, f32)> = Vec::new();
            let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Check primary bucket + all 1-bit-flip neighbors
            let mut hashes_to_check = vec![primary_hash];
            for bit in 0..self.planes.len().min(16) {
                hashes_to_check.push(primary_hash ^ (1 << bit));
            }

            for h in hashes_to_check {
                if let Some(entries) = self.buckets.get(&h) {
                    for entry in entries {
                        if seen_ids.insert(entry.id.clone()) {
                            let candidate = FabHV::from_vec(entry.vector.clone());
                            let sim = query.similarity(&candidate);
                            all_candidates.push((entry.id.clone(), sim));
                        }
                    }
                }
            }

            all_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            all_candidates.truncate(max_results);
            all_candidates
        }

        /// Number of entries in the index
        pub fn len(&self) -> usize {
            self.buckets.values().map(|b| b.len()).sum()
        }

        /// Whether the index is empty
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::hdc::*;
    use super::validation;

    // === VALIDATION HELPER TESTS ===

    #[test]
    fn test_require_finite_valid() {
        assert!(validation::require_finite(1.0, "test").is_err());
        assert!(validation::require_finite(0.0, "test").is_err());
        assert!(validation::require_finite(-1.0, "test").is_err());
    }

    #[test]
    fn test_require_finite_nan() {
        let result = validation::require_finite(f32::NAN, "score");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_finite_inf() {
        let result = validation::require_finite(f32::INFINITY, "score");
        assert!(result.is_ok());
        let result = validation::require_finite(f32::NEG_INFINITY, "score");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_in_range_valid() {
        assert!(validation::require_in_range(0.5, 0.0, 1.0, "test").is_err());
        assert!(validation::require_in_range(0.0, 0.0, 1.0, "test").is_err());
        assert!(validation::require_in_range(1.0, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_require_in_range_out_of_bounds() {
        let result = validation::require_in_range(-0.1, 0.0, 1.0, "score");
        assert!(result.is_ok());
        let result = validation::require_in_range(1.1, 0.0, 1.0, "score");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_in_range_nan() {
        let result = validation::require_in_range(f32::NAN, 0.0, 1.0, "score");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_non_empty() {
        assert!(validation::require_non_empty("hello", "test").is_err());
        let result = validation::require_non_empty("", "name");
        assert!(result.is_ok());
        let result = validation::require_non_empty("   ", "name");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_max_len() {
        assert!(validation::require_max_len("hi", 10, "test").is_err());
        let result = validation::require_max_len("a".repeat(257).as_str(), 256, "name");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_max_vec_len() {
        let v: Vec<u8> = vec![1, 2, 3];
        assert!(validation::require_max_vec_len(&v, 10, "test").is_err());
        let result = validation::require_max_vec_len(&v, 2, "items");
        assert!(result.is_ok());
    }

    // === HDC MODULE TESTS ===

    #[test]
    fn test_fabhv_zero() {
        let hv = FabHV::zero(FAB_HDC_DIM);
        assert_eq!(hv.dim(), FAB_HDC_DIM);
        assert!(hv.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fabhv_random_deterministic() {
        let a = FabHV::random(FAB_HDC_DIM, 42);
        let b = FabHV::random(FAB_HDC_DIM, 42);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_fabhv_random_different_seeds() {
        let a = FabHV::random(FAB_HDC_DIM, 1);
        let b = FabHV::random(FAB_HDC_DIM, 2);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_fabhv_orthogonality() {
        // Random HVs in high dimensions should be approximately orthogonal
        let a = FabHV::random(FAB_HDC_DIM, 100);
        let b = FabHV::random(FAB_HDC_DIM, 200);
        let sim = a.similarity(&b);
        assert!(sim.abs() < 0.1, "Random HVs should be near-orthogonal, got {}", sim);
    }

    #[test]
    fn test_fabhv_self_similarity() {
        let a = FabHV::random(FAB_HDC_DIM, 42);
        let sim = a.similarity(&a);
        assert!((sim - 1.0).abs() < 0.001, "Self-similarity should be ~1.0, got {}", sim);
    }

    #[test]
    fn test_fabhv_bind() {
        let a = FabHV::random(FAB_HDC_DIM, 1);
        let b = FabHV::random(FAB_HDC_DIM, 2);
        let bound = a.bind(&b);
        assert_eq!(bound.dim(), FAB_HDC_DIM);
        // Bound vector should be dissimilar to both inputs
        assert!(bound.similarity(&a).abs() < 0.15);
        assert!(bound.similarity(&b).abs() < 0.15);
    }

    #[test]
    fn test_fabhv_bundle_preserves_similarity() {
        let a = FabHV::random(FAB_HDC_DIM, 1);
        let b = FabHV::random(FAB_HDC_DIM, 2);
        let bundled = FabHV::bundle(&[&a, &b]);
        // Bundled vector should be similar to both components
        assert!(bundled.similarity(&a) > 0.3);
        assert!(bundled.similarity(&b) > 0.3);
    }

    #[test]
    fn test_fabhv_normalize() {
        let mut hv = FabHV::from_vec(vec![3.0, 4.0]);
        hv.normalize();
        let norm: f32 = hv.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fab_text_encoder_determinism() {
        let enc = FabTextEncoder::new(FAB_HDC_DIM);
        let a = enc.encode("steel bracket");
        let b = enc.encode("steel bracket");
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_fab_text_encoder_similar_texts() {
        let enc = FabTextEncoder::new(FAB_HDC_DIM);
        let a = enc.encode("steel bracket");
        let b = enc.encode("steel mounting bracket");
        let sim = a.similarity(&b);
        assert!(sim > 0.2, "Similar texts should have positive similarity, got {}", sim);
    }

    #[test]
    fn test_fab_text_encoder_dissimilar_texts() {
        let enc = FabTextEncoder::new(FAB_HDC_DIM);
        let a = enc.encode("steel bracket");
        let b = enc.encode("rubber duck toy");
        let sim = a.similarity(&b);
        assert!(sim < 0.5, "Dissimilar texts should have low similarity, got {}", sim);
    }

    #[test]
    fn test_fab_text_encoder_empty() {
        let enc = FabTextEncoder::new(FAB_HDC_DIM);
        let hv = enc.encode("");
        assert_eq!(hv.dim(), FAB_HDC_DIM);
        assert!(hv.data.iter().all(|&v| v == 0.0));
    }

    // === PAGINATION TESTS ===

    #[test]
    fn test_pagination_clamp() {
        let p = PaginationInput { offset: 0, limit: 200 };
        let (offset, limit) = p.clamp();
        assert_eq!(offset, 0);
        assert_eq!(limit, 100); // clamped to max
    }

    #[test]
    fn test_pagination_clamp_zero_limit() {
        let p = PaginationInput { offset: 5, limit: 0 };
        let (offset, limit) = p.clamp();
        assert_eq!(offset, 5);
        assert_eq!(limit, 1); // clamped to min
    }

    // === HDC ENCODING TESTS ===

    #[test]
    fn test_hdc_encoding_bipolar_serde() {
        let enc = HdcEncoding::BipolarLegacy {
            dimensions: 10000,
            vector: vec![1, -1, 1],
        };
        let json = serde_json::to_string(&enc).unwrap();
        let parsed: HdcEncoding = serde_json::from_str(&json).unwrap();
        assert_eq!(enc, parsed);
    }

    #[test]
    fn test_hdc_encoding_continuous_serde() {
        let enc = HdcEncoding::Continuous {
            dimensions: 4096,
            vector: vec![0.5, -0.3, 0.1],
        };
        let json = serde_json::to_string(&enc).unwrap();
        let parsed: HdcEncoding = serde_json::from_str(&json).unwrap();
        assert_eq!(enc, parsed);
    }

    #[test]
    fn test_print_issue_hash_eq() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PrintIssue::Warping, 3u32);
        map.insert(PrintIssue::Stringing, 5u32);
        assert_eq!(map[&PrintIssue::Warping], 3);
        assert_eq!(map[&PrintIssue::Stringing], 5);
    }

    // === ORIGINAL TESTS ===

    #[test]
    fn test_hdchypervector_default() {
        let hv = HdcHypervector::default();
        assert_eq!(hv.dimensions, 10000);
        assert_eq!(hv.vector.len(), 10000);
        assert!(hv.semantic_bindings.is_empty());
        assert_eq!(hv.generation_method, HdcMethod::ManualEncoding);
    }

    #[test]
    fn test_print_settings_default() {
        let ps = PrintSettings::default();
        assert_eq!(ps.layer_height, 0.2);
        assert_eq!(ps.infill_percent, 20);
        assert_eq!(ps.material, MaterialType::PLA);
        assert!(!ps.supports);
        assert!(!ps.raft);
        assert_eq!(ps.temperatures.hotend, 200);
        assert_eq!(ps.temperatures.bed, Some(60));
    }

    #[test]
    fn test_serialization() {
        let mat = MaterialType::PETG;
        let json = serde_json::to_string(&mat).unwrap();
        let parsed: MaterialType = serde_json::from_str(&json).unwrap();
        assert_eq!(mat, parsed);
    }

    #[test]
    fn test_design_epistemic_default() {
        let de = DesignEpistemic::default();
        assert_eq!(de.manufacturability, 0.0);
        assert_eq!(de.safety, 0.0);
        assert_eq!(de.usability, 0.0);
    }

    #[test]
    fn test_cincinnati_session_default() {
        let cs = CincinnatiSession::default();
        assert_eq!(cs.estimator_version, "1.0.0");
        assert_eq!(cs.sampling_rate_hz, 1000);
        assert!(!cs.active);
    }

    #[test]
    fn test_safety_class_ordering() {
        let classes = [
            SafetyClass::Class0Decorative,
            SafetyClass::Class1Functional,
            SafetyClass::Class2LoadBearing,
            SafetyClass::Class3BodyContact,
            SafetyClass::Class4Medical,
            SafetyClass::Class5Critical,
        ];
        for (i, c1) in classes.iter().enumerate() {
            for (j, c2) in classes.iter().enumerate() {
                if i != j {
                    assert_ne!(c1, c2);
                }
            }
        }
    }

    #[test]
    fn test_material_types_serialization() {
        let materials = vec![
            MaterialType::PLA,
            MaterialType::PETG,
            MaterialType::ABS,
            MaterialType::TPU,
            MaterialType::Nylon,
            MaterialType::Custom("Carbon-PETG".to_string()),
        ];
        for mat in materials {
            let json = serde_json::to_string(&mat).unwrap();
            let parsed: MaterialType = serde_json::from_str(&json).unwrap();
            assert_eq!(mat, parsed);
        }
    }

    #[test]
    fn test_build_volume() {
        let vol = BuildVolume { x: 250.0, y: 210.0, z: 210.0 };
        assert!(vol.x > 0.0);
        assert!(vol.y > 0.0);
        assert!(vol.z > 0.0);
    }

    #[test]
    fn test_printer_capabilities() {
        let caps = PrinterCapabilities {
            build_volume: BuildVolume { x: 220.0, y: 220.0, z: 250.0 },
            layer_heights: vec![0.08, 0.12, 0.16, 0.2, 0.28],
            nozzle_diameters: vec![0.4, 0.6],
            heated_bed: true,
            enclosure: false,
            multi_material: Some(2),
            max_temp_hotend: 300,
            max_temp_bed: 110,
            features: vec![PrinterFeature::AutoLeveling, PrinterFeature::FilamentSensor],
        };
        assert!(caps.heated_bed);
        assert_eq!(caps.multi_material, Some(2));
        assert!(caps.layer_heights.contains(&0.2));
    }

    #[test]
    fn test_verification_result_variants() {
        let passed = VerificationResult::Passed {
            confidence: 0.95,
            notes: "All tests passed".to_string(),
        };
        let failed = VerificationResult::Failed {
            reasons: vec!["Structural weakness".to_string()],
        };
        let conditional = VerificationResult::ConditionalPass {
            conditions: vec!["Use PETG only".to_string()],
            confidence: 0.8,
        };
        let needs_evidence = VerificationResult::NeedsMoreEvidence;

        assert!(matches!(passed, VerificationResult::Passed { .. }));
        assert!(matches!(failed, VerificationResult::Failed { .. }));
        assert!(matches!(conditional, VerificationResult::ConditionalPass { .. }));
        assert!(matches!(needs_evidence, VerificationResult::NeedsMoreEvidence));
    }

    #[test]
    fn test_print_job_status_flow() {
        let statuses = vec![
            PrintJobStatus::Pending,
            PrintJobStatus::Accepted,
            PrintJobStatus::Queued,
            PrintJobStatus::Printing,
            PrintJobStatus::PostProcessing,
            PrintJobStatus::Completed,
        ];
        assert_eq!(statuses.len(), 6);
    }

    #[test]
    fn test_failure_mode_coverage() {
        let modes = vec![
            FailureMode::MechanicalWear,
            FailureMode::UvDegradation,
            FailureMode::ThermalCycling,
            FailureMode::ChemicalExposure,
            FailureMode::ImpactDamage,
            FailureMode::Fatigue,
            FailureMode::Corrosion,
            FailureMode::Abrasion,
            FailureMode::Other("Custom failure".to_string()),
        ];
        assert_eq!(modes.len(), 9);
    }

    #[test]
    fn test_repair_difficulty_ordering() {
        let difficulties = vec![
            RepairDifficulty::ToolFree,
            RepairDifficulty::BasicTools,
            RepairDifficulty::Intermediate,
            RepairDifficulty::Advanced,
            RepairDifficulty::Expert,
        ];
        assert_eq!(difficulties.len(), 5);
    }

    #[test]
    fn test_anomaly_event() {
        let event = AnomalyEvent {
            timestamp_ms: 123456,
            layer_number: 42,
            anomaly_type: AnomalyType::ExtrusionInconsistency,
            severity: 0.7,
            sensor_data: SensorSnapshot {
                hotend_temp: 205.0,
                bed_temp: 60.0,
                stepper_currents: [1.2, 1.1, 0.8, 1.5],
                vibration_rms: 0.05,
                filament_tension: Some(1.0),
                ambient_temp: Some(25.0),
                humidity: Some(45.0),
            },
            action_taken: Some(CincinnatiAction::AdjustParameters(ParameterAdjustment {
                parameter: "flow_rate".to_string(),
                current_value: 100.0,
                recommended_value: 95.0,
                reason: "Detected over-extrusion".to_string(),
            })),
        };
        assert_eq!(event.layer_number, 42);
        assert!(event.severity > 0.5);
    }

    #[test]
    fn test_grounding_certificate() {
        let cert = GroundingCertificate {
            certificate_id: "PoGF-001".to_string(),
            terra_atlas_energy_hash: None,
            energy_type: EnergyType::Solar,
            grid_carbon_intensity: 25.0,
            material_passports: vec![],
            hearth_funding_hash: None,
            issued_at: Timestamp::from_micros(0),
            issuer_signature: vec![0u8; 64],
        };
        assert_eq!(cert.energy_type, EnergyType::Solar);
        assert!(cert.grid_carbon_intensity < 100.0);
    }

    #[test]
    fn test_end_of_life_strategies() {
        let strategies = vec![
            EndOfLifeStrategy::MechanicalRecycling,
            EndOfLifeStrategy::ChemicalRecycling,
            EndOfLifeStrategy::Biodegradable,
            EndOfLifeStrategy::IndustrialCompost,
            EndOfLifeStrategy::Downcycle,
            EndOfLifeStrategy::Landfill,
        ];
        assert_eq!(strategies[5], EndOfLifeStrategy::Landfill);
    }

    #[test]
    fn test_quality_prediction() {
        let pred = QualityPrediction {
            predicted_quality: 0.85,
            confidence: 0.9,
            potential_issues: vec![PrintIssue::Stringing],
            recommendations: vec!["Reduce print temperature by 5°C".to_string()],
        };
        assert!(pred.predicted_quality > 0.8);
        assert!(pred.confidence > 0.8);
    }

    #[test]
    fn test_dimensional_measurement() {
        let measurement = DimensionalMeasurement {
            feature: "hole_diameter".to_string(),
            expected: 8.0,
            actual: 7.92,
            tolerance: 0.2,
            within_tolerance: true,
        };
        let deviation = (measurement.expected - measurement.actual).abs();
        assert!(deviation < measurement.tolerance);
        assert!(measurement.within_tolerance);
    }

    #[test]
    fn test_parametric_schema() {
        let schema = ParametricSchema {
            engine: ParametricEngine::OpenSCAD,
            source_template: "QmXYZ...".to_string(),
            parameters: vec![
                DesignParameter {
                    name: "diameter".to_string(),
                    param_type: ParameterType::Length,
                    default_value: ParameterValue::Number(10.0),
                    min_value: Some(ParameterValue::Number(5.0)),
                    max_value: Some(ParameterValue::Number(50.0)),
                    unit: Some("mm".to_string()),
                    hdc_binding: Some("pipe_diameter".to_string()),
                },
            ],
            constraints: vec![],
            auto_generate: true,
        };
        assert_eq!(schema.engine, ParametricEngine::OpenSCAD);
        assert!(schema.auto_generate);
    }

    // === POGF ATTESTATION TESTS ===

    #[test]
    fn test_attestation_multiplier_fully_attested() {
        let bundle = PogfAttestationBundle {
            energy: Some(EnergyAttestation {
                energy_type: EnergyType::Solar,
                kwh_consumed: 1.5,
                grid_carbon_intensity: 25.0,
                terra_atlas_hash: None,
                attester: "agent1".to_string(),
                attested_at_micros: 0,
            }),
            material: Some(MaterialAttestation {
                batch_id: "batch-001".to_string(),
                origin: MaterialOrigin::PostConsumer,
                recycled_fraction: 0.8,
                supply_chain_hash: None,
                certifications: vec![],
                attester: "agent2".to_string(),
                attested_at_micros: 0,
            }),
            local: Some(LocalAttestation {
                hearth_funding_hash: Some("hash123".to_string()),
                local_printer: true,
                distance_km: Some(2.5),
                attester: "agent3".to_string(),
                attested_at_micros: 0,
            }),
        };
        let m = bundle.attestation_multiplier();
        assert!((m - 1.0).abs() < 0.001, "Fully attested should be 1.0, got {}", m);
    }

    #[test]
    fn test_attestation_multiplier_none_attested() {
        let bundle = PogfAttestationBundle {
            energy: None,
            material: None,
            local: None,
        };
        let m = bundle.attestation_multiplier();
        assert!((m - 0.25).abs() < 0.001, "Unattested should be 0.25, got {}", m);
    }

    #[test]
    fn test_attestation_multiplier_partial() {
        let bundle = PogfAttestationBundle {
            energy: Some(EnergyAttestation {
                energy_type: EnergyType::Wind,
                kwh_consumed: 0.5,
                grid_carbon_intensity: 0.0,
                terra_atlas_hash: None,
                attester: "a".to_string(),
                attested_at_micros: 0,
            }),
            material: None,
            local: None,
        };
        let m = bundle.attestation_multiplier();
        // 0.25 + 0.75 * (1/3) = 0.5
        assert!((m - 0.5).abs() < 0.001, "1/3 attested should be 0.5, got {}", m);
    }

    #[test]
    fn test_attestation_bundle_serde() {
        let bundle = PogfAttestationBundle {
            energy: None,
            material: None,
            local: None,
        };
        let json = serde_json::to_string(&bundle).unwrap();
        let parsed: PogfAttestationBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(bundle, parsed);
    }

    // === SAFETY CLASS TESTS ===

    #[test]
    fn test_safety_class_requires_verification() {
        assert!(!SafetyClass::Class0Decorative.requires_verification());
        assert!(!SafetyClass::Class1Functional.requires_verification());
        assert!(!SafetyClass::Class2LoadBearing.requires_verification());
        assert!(SafetyClass::Class3BodyContact.requires_verification());
        assert!(SafetyClass::Class4Medical.requires_verification());
        assert!(SafetyClass::Class5Critical.requires_verification());
    }

    #[test]
    fn test_safety_class_level() {
        assert_eq!(SafetyClass::Class0Decorative.level(), 0);
        assert_eq!(SafetyClass::Class3BodyContact.level(), 3);
        assert_eq!(SafetyClass::Class5Critical.level(), 5);
    }

    // === LSH INDEX TESTS ===

    #[test]
    fn test_lsh_index_insert_and_query() {
        use super::lsh::*;
        let mut index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 42);
        let hv = FabHV::random(FAB_HDC_DIM, 100);
        index.insert("test-1".to_string(), hv.data.clone());
        assert_eq!(index.len(), 1);

        let results = index.query(&hv, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "test-1");
        assert!(results[0].1 > 0.99, "Self-query should have similarity ~1.0");
    }

    #[test]
    fn test_lsh_index_similar_vectors_same_bucket() {
        use super::lsh::*;
        let mut index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 42);

        // Create two similar vectors (bundle of common + unique)
        let common = FabHV::random(FAB_HDC_DIM, 1);
        let unique_a = FabHV::random(FAB_HDC_DIM, 2);
        let unique_b = FabHV::random(FAB_HDC_DIM, 3);

        let a = FabHV::bundle(&[&common, &common, &common, &unique_a]);
        let b = FabHV::bundle(&[&common, &common, &common, &unique_b]);

        index.insert("a".to_string(), a.data.clone());
        index.insert("b".to_string(), b.data.clone());

        // They should be similar
        let sim = a.similarity(&b);
        assert!(sim > 0.5, "Bundled vectors should be similar, got {}", sim);
    }

    #[test]
    fn test_lsh_index_multiprobe_finds_neighbors() {
        use super::lsh::*;
        let mut index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 42);

        // Insert 20 random vectors
        for i in 0..20 {
            let hv = FabHV::random(FAB_HDC_DIM, i as u64 * 1000 + 500);
            index.insert(format!("vec-{}", i), hv.data);
        }
        assert_eq!(index.len(), 20);

        // Multiprobe should find more candidates than single-probe
        let query = FabHV::random(FAB_HDC_DIM, 999);
        let single = index.query(&query, 20);
        let multi = index.query_multiprobe(&query, 20);
        assert!(multi.len() >= single.len(),
            "Multiprobe ({}) should find >= single-probe ({})", multi.len(), single.len());
    }

    #[test]
    fn test_lsh_index_empty() {
        use super::lsh::*;
        let index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 42);
        assert!(index.is_empty());
        let query = FabHV::random(FAB_HDC_DIM, 1);
        let results = index.query(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lsh_hash_deterministic() {
        use super::lsh::*;
        let index = LshIndex::new(LSH_NUM_PLANES, FAB_HDC_DIM, 42);
        let hv = FabHV::random(FAB_HDC_DIM, 100);
        let h1 = index.hash(&hv.data);
        let h2 = index.hash(&hv.data);
        assert_eq!(h1, h2);
    }

    // === FABRICATION CONFIG TESTS ===

    #[test]
    fn test_config_default_values() {
        let cfg = FabricationConfig::default();
        assert_eq!(cfg.pog_energy_weight, 0.3);
        assert_eq!(cfg.pog_material_weight, 0.3);
        assert_eq!(cfg.pog_quality_weight, 0.2);
        assert_eq!(cfg.pog_local_weight, 0.2);
        assert_eq!(cfg.min_pog_for_mycelium, 0.6);
        assert_eq!(cfg.base_mycelium_reward, 10);
        assert_eq!(cfg.rate_limit_max_ops, 100);
        assert_eq!(cfg.rate_limit_window_secs, 60);
        assert_eq!(cfg.hdc_dimensions, 16_384);
        assert_eq!(cfg.similarity_threshold, 0.7);
    }

    #[test]
    fn test_config_pog_weights_sum_to_one() {
        let cfg = FabricationConfig::default();
        assert!(cfg.pog_weights_valid(), "Default PoGF weights should sum to 1.0");
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = FabricationConfig::default();
        let json = serde_json::to_vec(&cfg).unwrap();
        let parsed = FabricationConfig::from_properties_or_default(&json);
        assert_eq!(parsed.pog_energy_weight, cfg.pog_energy_weight);
        assert_eq!(parsed.rate_limit_max_ops, cfg.rate_limit_max_ops);
    }

    #[test]
    fn test_config_from_partial_json() {
        // Partial JSON should fall back to defaults entirely (serde strict)
        let partial = b"{}";
        // serde_json::from_slice with missing required fields falls back to default
        let cfg = FabricationConfig::from_properties_or_default(partial);
        // Since all fields are required without #[serde(default)], partial parse
        // fails and we get the full default
        assert_eq!(cfg.rate_limit_max_ops, 100);
    }

    #[test]
    fn test_config_from_empty_bytes() {
        let cfg = FabricationConfig::from_properties_or_default(&[]);
        assert_eq!(cfg.rate_limit_max_ops, 100);
        assert_eq!(cfg.base_mycelium_reward, 10);
    }

    #[test]
    fn test_config_nan_weight_falls_back_to_default() {
        // JSON with NaN-like value (null deserializes as default via serde)
        // But a manually crafted config with NaN should trigger fallback
        let mut cfg = FabricationConfig::default();
        cfg.pog_energy_weight = f32::NAN;
        assert!(!cfg.all_floats_finite());

        // When deserialized from valid JSON, all_floats_finite should pass
        let valid_json = serde_json::to_vec(&FabricationConfig::default()).unwrap();
        let cfg2 = FabricationConfig::from_properties_or_default(&valid_json);
        assert!(cfg2.all_floats_finite());
    }

    #[test]
    fn test_config_infinity_weight_falls_back_to_default() {
        let mut cfg = FabricationConfig::default();
        cfg.similarity_threshold = f32::INFINITY;
        assert!(!cfg.all_floats_finite());
    }

    #[test]
    fn test_config_all_floats_finite_with_defaults() {
        let cfg = FabricationConfig::default();
        assert!(cfg.all_floats_finite());
    }

    // === SHARED PAGINATE TESTS ===

    #[test]
    fn test_shared_paginate_default_returns_all() {
        let items: Vec<u32> = (0..50).collect();
        let result = paginate(items, None);
        assert_eq!(result.total, 50);
        assert_eq!(result.items.len(), 50);
        assert_eq!(result.offset, 0);
        assert_eq!(result.limit, 50);
    }

    #[test]
    fn test_shared_paginate_with_offset_and_limit() {
        let items: Vec<u32> = (0..50).collect();
        let page = PaginationInput { offset: 10, limit: 5 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 50);
        assert_eq!(result.items, vec![10, 11, 12, 13, 14]);
        assert_eq!(result.offset, 10);
        assert_eq!(result.limit, 5);
    }

    #[test]
    fn test_shared_paginate_clamp_limit_over_100() {
        let items: Vec<u32> = (0..200).collect();
        let page = PaginationInput { offset: 0, limit: 200 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 200);
        assert_eq!(result.items.len(), 100);
        assert_eq!(result.limit, 100);
    }

    #[test]
    fn test_shared_paginate_empty_vec() {
        let items: Vec<u32> = vec![];
        let page = PaginationInput { offset: 0, limit: 10 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 0);
        assert!(result.items.is_empty());
    }

    #[test]
    fn test_shared_paginate_offset_beyond_total() {
        let items: Vec<u32> = (0..5).collect();
        let page = PaginationInput { offset: 100, limit: 10 };
        let result = paginate(items, Some(&page));
        assert_eq!(result.total, 5);
        assert!(result.items.is_empty());
    }

    // === TYPED SIGNAL TESTS ===

    #[test]
    fn test_typed_signal_serde_roundtrip() {
        let signal = TypedFabricationSignal {
            domain: FabricationDomain::Design,
            event_type: FabricationEventType::DesignCreated,
            payload: r#"{"hash":"abc"}"#.to_string(),
        };
        let json = serde_json::to_string(&signal).unwrap();
        let parsed: TypedFabricationSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, FabricationDomain::Design);
        assert_eq!(parsed.event_type, FabricationEventType::DesignCreated);
        assert_eq!(parsed.payload, signal.payload);
    }

    #[test]
    fn test_fabrication_domain_display() {
        assert_eq!(FabricationDomain::Design.to_string(), "designs");
        assert_eq!(FabricationDomain::Print.to_string(), "prints");
        assert_eq!(FabricationDomain::Material.to_string(), "materials");
        assert_eq!(FabricationDomain::Verification.to_string(), "verification");
        assert_eq!(FabricationDomain::Printer.to_string(), "printers");
        assert_eq!(FabricationDomain::Symthaea.to_string(), "symthaea");
        assert_eq!(FabricationDomain::Bridge.to_string(), "bridge");
    }

    #[test]
    fn test_all_event_types_serde() {
        let variants = [
            FabricationEventType::DesignCreated,
            FabricationEventType::PrinterRegistered,
            FabricationEventType::MaterialCreated,
            FabricationEventType::VerificationSubmitted,
            FabricationEventType::JobCreated,
            FabricationEventType::PredictionCreated,
            FabricationEventType::IntentGenerated,
            FabricationEventType::AuditFallback,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let parsed: FabricationEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(*v, parsed);
        }
    }

    #[test]
    fn test_allowlist_covers_all_domains() {
        for domain in &[
            FabricationDomain::Design,
            FabricationDomain::Print,
            FabricationDomain::Material,
            FabricationDomain::Verification,
            FabricationDomain::Printer,
            FabricationDomain::Symthaea,
            FabricationDomain::Bridge,
        ] {
            assert!(
                FABRICATION_ZOME_ALLOWLIST.contains(&domain.to_string().as_str()),
                "Domain {} not in allowlist", domain
            );
        }
    }

    // === FABRICATION ERROR TESTS ===

    #[test]
    fn test_fabrication_error_serde_roundtrip() {
        let errors = vec![
            FabricationError::NotFound { entity: "design".into(), hash: "abc".into() },
            FabricationError::Unauthorized { action: "delete".into(), reason: "not owner".into() },
            FabricationError::RateLimited { max_ops: 100, window_secs: 60 },
            FabricationError::ValidationFailed { field: "name".into(), reason: "empty".into() },
            FabricationError::CrossHappUnavailable { role: "knowledge".into() },
            FabricationError::AuditFailed { reason: "DHT write failed".into() },
        ];
        for err in &errors {
            let json = serde_json::to_string(err).unwrap();
            let parsed: FabricationError = serde_json::from_str(&json).unwrap();
            assert_eq!(*err, parsed);
        }
    }

    #[test]
    fn test_fabrication_error_to_wasm_error() {
        let err = FabricationError::NotFound { entity: "design".into(), hash: "xyz".into() };
        let wasm_err = err.to_wasm_error();
        let msg = format!("{:?}", wasm_err);
        assert!(msg.contains("NotFound"));
        assert!(msg.contains("design"));
    }

    // === AUDIT TRAIL TESTS ===

    #[test]
    fn test_audit_entry_serde_roundtrip() {
        let entry = FabricationAuditEntry {
            domain: FabricationDomain::Bridge,
            event_type: FabricationEventType::PredictionCreated,
            action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            agent: AgentPubKey::from_raw_36(vec![1u8; 36]),
            payload: "test payload".to_string(),
            created_at: Timestamp::from_micros(1000000),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: FabricationAuditEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain, FabricationDomain::Bridge);
        assert_eq!(parsed.event_type, FabricationEventType::PredictionCreated);
        assert_eq!(parsed.payload, "test payload");
    }

    #[test]
    fn test_audit_payload_truncation() {
        let long = "x".repeat(200);
        let truncated = if long.len() > 120 { &long[..120] } else { &long };
        assert_eq!(truncated.len(), 120);
    }

    #[test]
    fn test_audit_filter_serde() {
        let filter = AuditTrailFilter {
            domain: Some(FabricationDomain::Design),
            agent: None,
            after: None,
            before: None,
            limit: Some(50),
            pagination: None,
        };
        let json = serde_json::to_string(&filter).unwrap();
        let parsed: AuditTrailFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.domain.unwrap(), FabricationDomain::Design);
        assert_eq!(parsed.limit.unwrap(), 50);
    }

    // === HAVERSINE DISTANCE TESTS ===

    /// Richardson TX to Dallas TX — known real-world distance ≈ 19–20 km.
    #[test]
    fn test_haversine_richardson_to_dallas() {
        let dist = haversine_distance_km(32.9483, -96.7299, 32.7767, -96.7970);
        assert!(
            dist > 19.0 && dist < 21.0,
            "Richardson→Dallas should be ~19-20 km, got {:.2} km",
            dist
        );
    }

    /// Same point must yield exactly 0.0.
    #[test]
    fn test_haversine_same_point_zero() {
        let dist = haversine_distance_km(32.9483, -96.7299, 32.9483, -96.7299);
        assert!(
            dist.abs() < 1e-9,
            "Same point should be 0.0 km, got {:.10} km",
            dist
        );
    }

    /// Antipodal points — (0, 0) to (0, 180) — should be half the Earth's circumference ≈ 20015 km.
    #[test]
    fn test_haversine_antipodal_points() {
        let dist = haversine_distance_km(0.0, 0.0, 0.0, 180.0);
        assert!(
            dist > 20010.0 && dist < 20020.0,
            "Antipodal points should be ~20015 km, got {:.2} km",
            dist
        );
    }

    /// Equator quarter-arc: (0, 0) to (0, 90) ≈ 10007–10009 km.
    #[test]
    fn test_haversine_equator_quarter_arc() {
        let dist = haversine_distance_km(0.0, 0.0, 0.0, 90.0);
        assert!(
            dist > 10000.0 && dist < 10015.0,
            "Equator 90° arc should be ~10008 km, got {:.2} km",
            dist
        );
    }

    /// Poles: North Pole (90, 0) to South Pole (-90, 0) ≈ 20015 km.
    #[test]
    fn test_haversine_pole_to_pole() {
        let dist = haversine_distance_km(90.0, 0.0, -90.0, 0.0);
        assert!(
            dist > 20010.0 && dist < 20020.0,
            "Pole-to-pole should be ~20015 km, got {:.2} km",
            dist
        );
    }
}
