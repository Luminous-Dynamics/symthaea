// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fabrication Common - Shared Types for Mycelix Fabrication hApp
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
    /// City name
    pub city: Option<String>,
    /// Region/State/Province
    pub region: Option<String>,
    /// Country code (ISO 3166-1 alpha-2)
    pub country: String,
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
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // Safety classes should be distinct
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
        // Test all material types serialize correctly
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
        let vol = BuildVolume {
            x: 250.0,
            y: 210.0,
            z: 210.0,
        };
        assert!(vol.x > 0.0);
        assert!(vol.y > 0.0);
        assert!(vol.z > 0.0);
    }

    #[test]
    fn test_printer_capabilities() {
        let caps = PrinterCapabilities {
            build_volume: BuildVolume {
                x: 220.0,
                y: 220.0,
                z: 250.0,
            },
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
        assert!(matches!(
            conditional,
            VerificationResult::ConditionalPass { .. }
        ));
        assert!(matches!(
            needs_evidence,
            VerificationResult::NeedsMoreEvidence
        ));
    }

    #[test]
    fn test_print_job_status_flow() {
        // Typical status flow: Pending -> Accepted -> Queued -> Printing -> Completed
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
        assert!(cert.grid_carbon_intensity < 100.0); // Low carbon
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
        // Landfill should be discouraged (last resort)
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
            parameters: vec![DesignParameter {
                name: "diameter".to_string(),
                param_type: ParameterType::Length,
                default_value: ParameterValue::Number(10.0),
                min_value: Some(ParameterValue::Number(5.0)),
                max_value: Some(ParameterValue::Number(50.0)),
                unit: Some("mm".to_string()),
                hdc_binding: Some("pipe_diameter".to_string()),
            }],
            constraints: vec![],
            auto_generate: true,
        };
        assert_eq!(schema.engine, ParametricEngine::OpenSCAD);
        assert!(schema.auto_generate);
    }
}
