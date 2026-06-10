// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared types for the support sub-crate (framework-independent mirrors of zome types)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SupportCategory {
    Network,
    Hardware,
    Software,
    Holochain,
    Mycelix,
    Security,
    General,
}

impl SupportCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Network => "Network",
            Self::Hardware => "Hardware",
            Self::Software => "Software",
            Self::Holochain => "Holochain",
            Self::Mycelix => "Mycelix",
            Self::Security => "Security",
            Self::General => "General",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TicketPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TicketStatus {
    Open,
    InProgress,
    AwaitingUser,
    Resolved,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AutonomyLevel {
    Advisory,
    SemiAutonomous,
    FullAutonomous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionType {
    RestartService,
    ClearCache,
    UpdateConfig,
    RunDiagnostic,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiagnosticType {
    NetworkCheck,
    DiskSpace,
    ServiceStatus,
    HolochainHealth,
    MemoryUsage,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Healthy,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicStatus {
    Certain,
    Probable,
    Uncertain,
    Unknown,
    OutOfDomain,
}

/// Triage result from the triage engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageResult {
    pub suggested_priority: TicketPriority,
    pub suggested_category: SupportCategory,
    pub similar_tickets: Vec<SimilarTicket>,
    pub confidence: f32,
    pub phi: f64,
    pub suggested_articles: Vec<String>,
    pub epistemic_status: EpistemicStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarTicket {
    pub id: String,
    pub title: String,
    pub similarity: f32,
}

/// Diagnostic plan from the diagnostic planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticPlan {
    pub steps: Vec<DiagnosticStep>,
    pub expected_resolution_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticStep {
    pub diagnostic_type: DiagnosticType,
    pub description: String,
    pub expected_info_gain: f64,
    pub autonomous_capable: bool,
}

/// Knowledge graduation decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraduationDecision {
    Graduate,
    Defer(String),
    Reject(String),
}

/// Seed article for knowledge pre-seeding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedArticle {
    pub title: String,
    pub content: String,
    pub category: SupportCategory,
    pub tags: Vec<String>,
}

/// Scrub result from the privacy scrubber.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubResult {
    pub scrubbed_text: String,
    pub redaction_count: usize,
    pub redaction_types: std::collections::HashMap<String, usize>,
}

/// Proposed action from the action engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    pub action_type: ActionType,
    pub description: String,
    pub rollback_steps: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub success: bool,
    pub message: String,
    pub state_snapshot: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackResult {
    pub success: bool,
    pub message: String,
}

/// System telemetry for the predictive engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTelemetry {
    pub conductor_memory_mb: f64,
    pub gossip_peers: usize,
    pub disk_free_pct: f64,
    pub network_latency_ms: f64,
    pub error_rate_per_min: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub expected_free_energy: f64,
    pub predicted_failure: Option<String>,
    pub time_horizon_secs: Option<u64>,
    pub confidence: f32,
}

/// Knowledge search match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeMatch {
    pub article_id: String,
    pub title: String,
    pub similarity: f32,
    pub phi: f64,
}

/// Cognitive update for federated learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveUpdateData {
    pub category: SupportCategory,
    pub encoding: Vec<u8>,
    pub phi: f64,
    pub resolution_pattern: String,
}

/// Absorption result for cognitive update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsorptionResult {
    pub absorbed: bool,
    pub similarity_to_existing: f32,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_triage_result() {
        let result = TriageResult {
            suggested_priority: TicketPriority::High,
            suggested_category: SupportCategory::Network,
            similar_tickets: vec![SimilarTicket {
                id: "T-1".to_string(),
                title: "DNS failure".to_string(),
                similarity: 0.85,
            }],
            confidence: 0.9,
            phi: 0.42,
            suggested_articles: vec!["KB-001".to_string()],
            epistemic_status: EpistemicStatus::Probable,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: TriageResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.suggested_priority, TicketPriority::High);
        assert_eq!(back.suggested_category, SupportCategory::Network);
        assert_eq!(back.similar_tickets.len(), 1);
        assert!((back.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn serde_roundtrip_diagnostic_plan() {
        let plan = DiagnosticPlan {
            steps: vec![DiagnosticStep {
                diagnostic_type: DiagnosticType::NetworkCheck,
                description: "Check DNS".to_string(),
                expected_info_gain: 0.9,
                autonomous_capable: true,
            }],
            expected_resolution_confidence: 0.75,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let back: DiagnosticPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back.steps.len(), 1);
        assert_eq!(back.steps[0].diagnostic_type, DiagnosticType::NetworkCheck);
    }

    #[test]
    fn serde_roundtrip_scrub_result() {
        let mut redaction_types = std::collections::HashMap::new();
        redaction_types.insert("ip".to_string(), 2);
        let result = ScrubResult {
            scrubbed_text: "Redacted text".to_string(),
            redaction_count: 2,
            redaction_types,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ScrubResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.redaction_count, 2);
        assert_eq!(back.redaction_types["ip"], 2);
    }

    #[test]
    fn serde_roundtrip_proposed_action() {
        let action = ProposedAction {
            action_type: ActionType::RestartService,
            description: "Restart conductor".to_string(),
            rollback_steps: vec!["Check status".to_string()],
            confidence: 0.8,
        };
        let json = serde_json::to_string(&action).unwrap();
        let back: ProposedAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back.action_type, ActionType::RestartService);
        assert_eq!(back.rollback_steps.len(), 1);
    }

    #[test]
    fn serde_roundtrip_system_telemetry() {
        let telemetry = SystemTelemetry {
            conductor_memory_mb: 2048.0,
            gossip_peers: 5,
            disk_free_pct: 45.0,
            network_latency_ms: 12.5,
            error_rate_per_min: 0.1,
        };
        let json = serde_json::to_string(&telemetry).unwrap();
        let back: SystemTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.gossip_peers, 5);
        assert!((back.conductor_memory_mb - 2048.0).abs() < f64::EPSILON);
    }

    #[test]
    fn serde_roundtrip_prediction_result() {
        let result = PredictionResult {
            expected_free_energy: 3.14,
            predicted_failure: Some("High memory".to_string()),
            time_horizon_secs: Some(3600),
            confidence: 0.7,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: PredictionResult = serde_json::from_str(&json).unwrap();
        assert!((back.expected_free_energy - 3.14).abs() < f64::EPSILON);
        assert_eq!(back.predicted_failure, Some("High memory".to_string()));
        assert_eq!(back.time_horizon_secs, Some(3600));
    }
}
