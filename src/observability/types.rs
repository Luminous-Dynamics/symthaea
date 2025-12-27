/*!
 * Event types for observability
 *
 * These types match the trace schema defined in tools/trace-schema-v1.json
 */

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Router selection event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterSelectionEvent {
    pub timestamp: DateTime<Utc>,
    pub input: String,
    pub selected_router: String,
    pub confidence: f64,
    pub alternatives: Vec<RouterAlternative>,
    pub bandit_stats: HashMap<String, BanditStats>,
}

impl Default for RouterSelectionEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            input: String::new(),
            selected_router: String::new(),
            confidence: 0.0,
            alternatives: Vec::new(),
            bandit_stats: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterAlternative {
    pub router: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditStats {
    pub count: u64,
    pub reward: f64,
}

/// Global Workspace Theory ignition event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceIgnitionEvent {
    pub timestamp: DateTime<Utc>,
    pub phi: f64,
    pub free_energy: f64,
    pub coalition_size: usize,
    pub active_primitives: Vec<String>,
    pub broadcast_payload_size: usize,
}

impl Default for WorkspaceIgnitionEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            phi: 0.0,
            free_energy: 0.0,
            coalition_size: 0,
            active_primitives: Vec::new(),
            broadcast_payload_size: 0,
        }
    }
}

/// Φ (Integrated Information) measurement event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhiMeasurementEvent {
    pub timestamp: DateTime<Utc>,
    pub phi: f64,
    pub components: PhiComponents,
    pub temporal_continuity: f64,
}

impl Default for PhiMeasurementEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            phi: 0.0,
            components: PhiComponents::default(),
            temporal_continuity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhiComponents {
    pub integration: f64,
    pub binding: f64,
    pub workspace: f64,
    pub attention: f64,
    pub recursion: f64,
    pub efficacy: f64,
    pub knowledge: f64,
}

/// Primitive activation event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PrimitiveActivationEvent {
    pub timestamp: DateTime<Utc>,
    pub primitive_name: String,
    pub tier: String,
    pub activation_strength: f64,
    pub context: Vec<String>,
}

impl Default for PrimitiveActivationEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            primitive_name: String::new(),
            tier: String::new(),
            activation_strength: 0.0,
            context: Vec::new(),
        }
    }
}

/// Response generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseGeneratedEvent {
    pub timestamp: DateTime<Utc>,
    pub content: String,
    pub confidence: f64,
    pub safety_verified: bool,
    pub requires_confirmation: bool,
    pub intent: String,
}

impl Default for ResponseGeneratedEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            content: String::new(),
            confidence: 0.0,
            safety_verified: false,
            requires_confirmation: false,
            intent: String::new(),
        }
    }
}

/// Security authorization check event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityCheckEvent {
    pub timestamp: DateTime<Utc>,
    pub operation: String,
    pub decision: SecurityDecision,
    pub reason: Option<String>,
    pub secrets_redacted: usize,
    /// Similarity score for pattern matching (0.0 - 1.0)
    pub similarity_score: Option<f64>,
    /// Pattern that was matched (if any)
    pub matched_pattern: Option<String>,
}

impl Default for SecurityCheckEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            operation: String::new(),
            decision: SecurityDecision::Allowed,
            reason: None,
            secrets_redacted: 0,
            similarity_score: None,
            matched_pattern: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityDecision {
    Allowed,
    Denied,
    RequiresConfirmation,
}

/// Error occurrence event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ErrorEvent {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub message: String,
    pub context: HashMap<String, String>,
    pub recoverable: bool,
}

impl Default for ErrorEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            error_type: String::new(),
            message: String::new(),
            context: HashMap::new(),
            recoverable: true,
        }
    }
}

/// Language understanding pipeline step event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LanguageStepEvent {
    pub timestamp: DateTime<Utc>,
    pub step_type: LanguageStepType,
    pub input: String,
    pub output: String,
    pub confidence: f64,
    pub duration_ms: u64,
}

impl Default for LanguageStepEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            step_type: LanguageStepType::Parsing,
            input: String::new(),
            output: String::new(),
            confidence: 0.0,
            duration_ms: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageStepType {
    Parsing,
    FrameExtraction,
    ConstructionMatching,
    SemanticEncoding,
    IntentRecognition,
    ResponseGeneration,
}

/// Trace format matching tools/trace-schema-v1.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub version: String,
    pub session_id: String,
    pub timestamp_start: DateTime<Utc>,
    pub timestamp_end: Option<DateTime<Utc>>,
    pub events: Vec<Event>,
    pub summary: Option<SessionSummary>,
}

impl Trace {
    pub fn new(session_id: String) -> Self {
        Self {
            version: "1.0".to_string(),
            session_id,
            timestamp_start: Utc::now(),
            timestamp_end: None,
            events: Vec::new(),
            summary: None,
        }
    }

    /// Load trace from JSON file
    pub fn load_from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let trace = serde_json::from_str(&content)?;
        Ok(trace)
    }
}

/// Generic event wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: serde_json::Value,
}

/// Session summary statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionSummary {
    pub total_events: usize,
    pub average_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub router_distribution: HashMap<String, usize>,
    pub ignition_count: usize,
    pub duration_ms: u64,
    pub errors: usize,
    pub security_denials: usize,
    /// **NEW**: Narrative Self statistics
    pub self_phi_average: f64,
    pub self_coherence_average: f64,
    pub veto_count: usize,
    /// **NEW**: Cross-Modal binding statistics
    pub cross_modal_phi_average: f64,
    pub binding_events: usize,
}

// ============================================================================
// REVOLUTIONARY IMPROVEMENT #73: Narrative Self + Cross-Modal Φ Tracing
// ============================================================================

/// Narrative Self state event - tracks identity coherence and veto decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NarrativeSelfEvent {
    pub timestamp: DateTime<Utc>,
    /// Self-Φ (identity integration measure)
    pub self_phi: f64,
    /// Overall coherence across all self levels
    pub coherence: f64,
    /// Proto-self coherence (immediate bodily state)
    pub proto_coherence: f64,
    /// Core-self coherence (persistent identity)
    pub core_coherence: f64,
    /// Autobiographical-self coherence (extended narrative)
    pub autobio_coherence: f64,
    /// Number of active goals
    pub active_goals: usize,
    /// Number of core values
    pub core_values: usize,
    /// Veto decision (if any)
    pub veto: Option<NarrativeSelfVeto>,
    /// Delta from previous Self-Φ
    pub phi_delta: f64,
    /// Whether a Φ-warning was triggered
    pub warning_triggered: bool,
}

impl Default for NarrativeSelfEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            self_phi: 0.0,
            coherence: 0.0,
            proto_coherence: 0.0,
            core_coherence: 0.0,
            autobio_coherence: 0.0,
            active_goals: 0,
            core_values: 0,
            veto: None,
            phi_delta: 0.0,
            warning_triggered: false,
        }
    }
}

/// Veto decision from Narrative Self
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSelfVeto {
    /// Whether the action was vetoed
    pub vetoed: bool,
    /// Reason category
    pub reason: String,
    /// Confidence in the veto decision
    pub confidence: f64,
    /// Projected impact on Self-Φ
    pub projected_phi_impact: f64,
    /// Description of the vetoed action
    pub action_description: Option<String>,
}

/// Cross-Modal Semantic Binding event - tracks multi-modal integration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CrossModalBindingEvent {
    pub timestamp: DateTime<Utc>,
    /// Cross-modal Φ (integration across modalities)
    pub cross_modal_phi: f64,
    /// Binding strength (similarity of bound representations)
    pub binding_strength: f64,
    /// Number of modalities involved
    pub modality_count: usize,
    /// Names of bound modalities
    pub modalities: Vec<String>,
    /// Number of features in the binding
    pub feature_count: usize,
    /// Binding duration in microseconds
    pub binding_duration_us: u64,
    /// Whether binding produced a unified percept
    pub unified_percept: bool,
    /// Semantic coherence of the binding
    pub semantic_coherence: f64,
}

impl Default for CrossModalBindingEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            cross_modal_phi: 0.0,
            binding_strength: 0.0,
            modality_count: 0,
            modalities: Vec::new(),
            feature_count: 0,
            binding_duration_us: 0,
            unified_percept: false,
            semantic_coherence: 0.0,
        }
    }
}

/// GWT Integration event - tracks Narrative Self in Global Workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GWTIntegrationEvent {
    pub timestamp: DateTime<Utc>,
    /// Ignition ID
    pub ignition_id: usize,
    /// Self-Φ before ignition
    pub self_phi_before: f64,
    /// Self-Φ after ignition
    pub self_phi_after: f64,
    /// Delta (change in Self-Φ)
    pub phi_delta: f64,
    /// Content that was broadcast
    pub broadcast_content: String,
    /// Goal alignment score for the broadcast
    pub goal_alignment: f64,
    /// Cross-modal Φ (if cross-modal binding was involved)
    pub cross_modal_phi: Option<f64>,
    /// Whether any content was vetoed
    pub vetoes_issued: usize,
}

impl Default for GWTIntegrationEvent {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            ignition_id: 0,
            self_phi_before: 0.0,
            self_phi_after: 0.0,
            phi_delta: 0.0,
            broadcast_content: String::new(),
            goal_alignment: 0.0,
            cross_modal_phi: None,
            vetoes_issued: 0,
        }
    }
}
