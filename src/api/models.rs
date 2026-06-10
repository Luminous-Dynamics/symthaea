// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API request and response models

use crate::control_plane::AuditEvent;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Submission request for model evaluation
#[derive(Debug, Clone, Deserialize)]
pub struct SubmissionRequest {
    pub model_name: String,
    pub topology_type: Option<TopologyType>,
    pub n_nodes: Option<usize>,
    pub dimension: Option<usize>,
    pub adjacency_matrix: Option<Vec<Vec<f32>>>,
    pub edge_list: Option<Vec<[usize; 2]>>,
    pub node_representations: Option<Vec<Vec<f32>>>,
    pub description: Option<String>,
    pub notify_email: Option<String>,
    pub public: Option<bool>,
}

/// Predefined topology types
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TopologyType {
    Ring,
    Torus,
    Hypercube,
    Star,
    Random,
    SmallWorld,
    Custom,
    Dense,
    Lattice,
    Line,
    BinaryTree,
    Modular,
    MobiusStrip,
    KleinBottle,
    Hyperbolic,
    ScaleFree,
    Fractal,
    Quantum,
}

/// Submission response
#[derive(Debug, Clone, Serialize)]
pub struct SubmissionResponse {
    pub submission_id: Uuid,
    pub status: SubmissionStatus,
    pub estimated_time_seconds: u32,
    pub position_in_queue: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Submission status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionStatus {
    Queued,
    Processing,
    Completed,
    Failed,
}

/// Processing status (for in-progress submissions)
#[derive(Debug, Clone, Serialize)]
pub struct ProcessingStatus {
    pub submission_id: Uuid,
    pub status: SubmissionStatus,
    pub progress_percent: f32,
    pub estimated_remaining_seconds: u32,
}

/// Evaluation result
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationResult {
    pub submission_id: Uuid,
    pub model_name: String,
    pub status: SubmissionStatus,
    /// Spectral connectivity (lambda2); legacy field name retained for compatibility.
    pub phi: f32,
    pub phi_confidence_interval: ConfidenceInterval,
    pub standard_deviation: f32,
    pub n_samples: u32,
    pub rank: u32,
    pub total_submissions: u32,
    pub percentile: f32,
    pub comparison_vs_baselines: std::collections::HashMap<String, BaselineComparison>,
    pub detailed_metrics: Option<DetailedMetrics>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub processing_time_seconds: Option<f32>,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceInterval {
    pub lower: f32,
    pub upper: f32,
    pub confidence_level: f32,
}

/// Baseline comparison
#[derive(Debug, Clone, Serialize)]
pub struct BaselineComparison {
    pub phi_difference: f32,
    pub percent_difference: f32,
    pub significantly_different: bool,
    pub p_value: f32,
}

/// Detailed metrics
#[derive(Debug, Clone, Serialize)]
pub struct DetailedMetrics {
    pub algebraic_connectivity: f32,
    pub spectral_gap: f32,
    pub eigenvalue_spectrum: Vec<f32>,
    pub similarity_matrix_stats: SimilarityMatrixStats,
}

/// Similarity matrix statistics
#[derive(Debug, Clone, Serialize)]
pub struct SimilarityMatrixStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

/// Leaderboard response
#[derive(Debug, Clone, Serialize)]
pub struct LeaderboardResponse {
    pub total_submissions: u32,
    pub page: u32,
    pub limit: u32,
    pub entries: Vec<LeaderboardEntry>,
}

/// Leaderboard entry
#[derive(Debug, Clone, Serialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub model_name: String,
    pub phi: f32,
    pub n_nodes: u32,
    pub topology_type: String,
    pub category: String,
    pub submitted_by: String,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
}

/// Topology rankings response
#[derive(Debug, Clone, Serialize)]
pub struct TopologyRankings {
    pub rankings: Vec<TopologyRankingEntry>,
}

/// Single topology ranking
#[derive(Debug, Clone, Serialize)]
pub struct TopologyRankingEntry {
    pub rank: u32,
    pub name: String,
    pub category: String,
    pub mean_phi: f32,
    pub std_dev: f32,
    pub n_samples: u32,
    pub vs_random: VsRandomComparison,
}

/// Comparison vs random baseline
#[derive(Debug, Clone, Serialize)]
pub struct VsRandomComparison {
    pub percent_difference: f32,
    pub significant: bool,
    pub p_value: f32,
}

/// Dataset list response
#[derive(Debug, Clone, Serialize)]
pub struct DatasetList {
    pub datasets: Vec<DatasetSummary>,
}

/// Dataset summary
#[derive(Debug, Clone, Serialize)]
pub struct DatasetSummary {
    pub id: String,
    pub name: String,
    pub category: String,
    pub size_mb: f32,
    pub n_samples: u32,
    pub description: String,
    pub license: String,
}

/// Full dataset info
#[derive(Debug, Clone, Serialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub version: String,
    pub size_mb: f32,
    pub n_samples: u32,
    pub license: String,
    pub citation: String,
    pub download_url: String,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Comparison request
#[derive(Debug, Clone, Deserialize)]
pub struct ComparisonRequest {
    pub model_a: ModelReference,
    pub model_b: ModelReference,
}

/// Model reference (either UUID or baseline name)
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ModelReference {
    SubmissionId(Uuid),
    BaselineName(String),
}

/// Comparison result
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    pub model_a: ModelInfo,
    pub model_b: ModelInfo,
    pub comparison: ComparisonDetails,
}

/// Model info for comparison
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub phi: f32,
    pub std_dev: f32,
}

/// Comparison details
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonDetails {
    pub absolute_difference: f32,
    pub percent_difference: f32,
    pub winner: String,
    pub statistically_significant: bool,
    pub t_statistic: f32,
    pub p_value: f32,
    pub effect_size_cohens_d: f32,
    pub interpretation: String,
}

/// Dimensional sweep request
#[derive(Debug, Clone, Deserialize)]
pub struct DimensionalSweepRequest {
    pub topology_type: TopologyType,
    pub min_dimension: Option<usize>,
    pub max_dimension: Option<usize>,
    pub samples_per_dimension: Option<usize>,
}

/// API error
#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Audit event list response
#[derive(Debug, Clone, Serialize)]
pub struct AuditEventsResponse {
    pub events: Vec<AuditEvent>,
}

impl ApiError {
    pub fn bad_request(message: &str) -> Self {
        Self {
            code: "INVALID_REQUEST".to_string(),
            message: message.to_string(),
            details: None,
        }
    }

    pub fn not_found(message: &str) -> Self {
        Self {
            code: "NOT_FOUND".to_string(),
            message: message.to_string(),
            details: None,
        }
    }

    pub fn internal_error(message: &str) -> Self {
        Self {
            code: "INTERNAL_ERROR".to_string(),
            message: message.to_string(),
            details: None,
        }
    }

    pub fn service_unavailable(message: &str) -> Self {
        Self {
            code: "SERVICE_UNAVAILABLE".to_string(),
            message: message.to_string(),
            details: None,
        }
    }
}
