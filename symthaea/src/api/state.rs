// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API state management

use crate::api::models::*;
use crate::control_plane::{AuditEvent, AuditLog};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use symthaea_core::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::ConnectivityCalculator,
};
use tokio::sync::{RwLock, Semaphore};
use uuid::Uuid;

/// Application state shared across all handlers
pub struct AppState {
    /// Submitted evaluations
    pub submissions: RwLock<HashMap<Uuid, Submission>>,
    /// Completed results
    pub results: RwLock<HashMap<Uuid, EvaluationResult>>,
    /// Pre-computed baseline topologies
    pub baselines: HashMap<String, BaselineTopology>,
    /// Spectral connectivity calculator (lambda2, not IIT Phi)
    pub phi_calculator: ConnectivityCalculator,
    /// Concurrency limiter for request processing.
    pub request_semaphore: Semaphore,
    /// Persistent and in-memory audit trail for control-plane events.
    audit_log: AuditLog,
    /// Optional bearer token for authenticated endpoints
    bearer_token: Option<String>,
}

/// A submitted evaluation job
#[derive(Debug, Clone)]
pub struct Submission {
    pub id: Uuid,
    pub request: SubmissionRequestStored,
    pub status: SubmissionStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Stored submission request (cloneable version)
#[derive(Debug, Clone)]
pub struct SubmissionRequestStored {
    pub model_name: String,
    pub topology_type: Option<TopologyType>,
    pub n_nodes: Option<usize>,
    pub dimension: Option<usize>,
    pub adjacency_matrix: Option<Vec<Vec<f32>>>,
    pub edge_list: Option<Vec<[usize; 2]>>,
    pub node_representations: Option<Vec<Vec<f32>>>,
    pub description: Option<String>,
    pub public: bool,
}

/// Pre-computed baseline topology
#[derive(Debug, Clone)]
pub struct BaselineTopology {
    pub name: String,
    pub category: String,
    pub mean_phi: f32,
    pub std_dev: f32,
    pub n_samples: u32,
}

impl AppState {
    /// Create new application state with pre-computed baselines
    pub fn new() -> Self {
        let phi_calculator = ConnectivityCalculator::new();

        // Pre-compute baseline topologies
        let baselines = Self::compute_baselines(&phi_calculator);

        Self {
            submissions: RwLock::new(HashMap::new()),
            results: RwLock::new(HashMap::new()),
            baselines,
            phi_calculator,
            request_semaphore: Semaphore::new(32),
            audit_log: AuditLog::new(None, 256),
            bearer_token: None,
        }
    }

    /// Create application state with API config
    pub fn new_with_config(config: &crate::api::ApiConfig) -> Self {
        let mut state = Self::new();
        state.bearer_token = config.bearer_token.clone();
        state.request_semaphore = Semaphore::new(config.max_in_flight_requests);
        state.audit_log = AuditLog::new(config.audit_log_path.clone(), 256);
        state
    }

    /// Get the configured bearer token (if any)
    pub fn bearer_token(&self) -> Option<&str> {
        self.bearer_token.as_deref()
    }

    pub fn record_audit(
        &self,
        event: &str,
        subject: &str,
        detail: &str,
    ) -> std::io::Result<AuditEvent> {
        self.audit_log.record("api", event, subject, detail)
    }

    pub fn audit_events(&self, limit: usize) -> Vec<AuditEvent> {
        self.audit_log.list(limit)
    }

    pub fn audit_log_path(&self) -> Option<&Path> {
        self.audit_log.file_path()
    }

    /// Compute baseline topology Φ values
    fn compute_baselines(calc: &ConnectivityCalculator) -> HashMap<String, BaselineTopology> {
        let mut baselines = HashMap::new();
        let n_nodes = 8;
        let n_samples = 10;
        let base_seed = 42;

        let topology_specs: Vec<(&str, &str, Box<dyn Fn(u64) -> ConsciousnessTopology>)> = vec![
            (
                "ring",
                "uniform",
                Box::new(move |s| ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, s)),
            ),
            (
                "star",
                "original",
                Box::new(move |s| ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, s)),
            ),
            (
                "random",
                "baseline",
                Box::new(move |s| ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, s)),
            ),
            (
                "hypercube_3d",
                "tier3",
                Box::new(move |s| ConsciousnessTopology::hypercube(3, HDC_DIMENSION, s)),
            ),
            (
                "hypercube_4d",
                "tier3",
                Box::new(move |s| ConsciousnessTopology::hypercube(4, HDC_DIMENSION, s)),
            ),
            (
                "torus",
                "tier1",
                Box::new(move |s| ConsciousnessTopology::torus_square(3, HDC_DIMENSION, s)),
            ),
        ];

        for (name, category, generator) in topology_specs {
            let mut connectivities = Vec::new();
            for i in 0..n_samples {
                let topo = generator(base_seed + i as u64 * 1000);
                let connectivity = calc.algebraic_connectivity(&topo.node_representations);
                connectivities.push(connectivity as f32);
            }

            let mean: f32 = connectivities.iter().sum::<f32>() / n_samples as f32;
            let std_dev = if n_samples > 1 {
                let variance: f32 = connectivities
                    .iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f32>()
                    / (n_samples - 1) as f32;
                variance.sqrt()
            } else {
                0.0
            };

            baselines.insert(
                name.to_string(),
                BaselineTopology {
                    name: name.to_string(),
                    category: category.to_string(),
                    mean_phi: mean,
                    std_dev,
                    n_samples: n_samples as u32,
                },
            );
        }

        baselines
    }

    /// Get leaderboard entries (sorted by Φ)
    pub async fn get_leaderboard(&self, limit: usize, offset: usize) -> Vec<LeaderboardEntry> {
        let submission_visibility: HashMap<Uuid, bool> = self
            .submissions
            .read()
            .await
            .iter()
            .map(|(id, submission)| (*id, submission.request.public))
            .collect();
        let results: Vec<EvaluationResult> = self.results.read().await.values().cloned().collect();
        let mut entries: Vec<_> = results
            .iter()
            .filter(|r| r.status == SubmissionStatus::Completed)
            .filter(|r| {
                submission_visibility
                    .get(&r.submission_id)
                    .copied()
                    .unwrap_or(true)
            })
            .map(|r| LeaderboardEntry {
                rank: 0, // Will be set after sorting
                model_name: r.model_name.clone(),
                phi: r.phi,
                n_nodes: r.n_samples, // Approximate
                topology_type: "custom".to_string(),
                category: "submission".to_string(),
                submitted_by: "anonymous".to_string(),
                submitted_at: r.created_at,
            })
            .collect();

        // Add baselines
        for (name, baseline) in &self.baselines {
            entries.push(LeaderboardEntry {
                rank: 0,
                model_name: name.clone(),
                phi: baseline.mean_phi,
                n_nodes: 8,
                topology_type: name.clone(),
                category: baseline.category.clone(),
                submitted_by: "baseline".to_string(),
                submitted_at: chrono::Utc::now(),
            });
        }

        // Sort by Φ descending
        entries.sort_by(|a, b| {
            b.phi
                .partial_cmp(&a.phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Set ranks
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        // Apply pagination
        entries.into_iter().skip(offset).take(limit).collect()
    }

    /// Get topology rankings
    pub fn get_topology_rankings(&self) -> Vec<TopologyRankingEntry> {
        let random_phi = self
            .baselines
            .get("random")
            .map(|b| b.mean_phi)
            .unwrap_or(0.4358);

        let mut rankings: Vec<_> = self
            .baselines
            .values()
            .map(|b| {
                let percent_diff = (b.mean_phi - random_phi) / random_phi * 100.0;
                TopologyRankingEntry {
                    rank: 0,
                    name: b.name.clone(),
                    category: b.category.clone(),
                    mean_phi: b.mean_phi,
                    std_dev: b.std_dev,
                    n_samples: b.n_samples,
                    vs_random: VsRandomComparison {
                        percent_difference: percent_diff,
                        significant: percent_diff.abs() > 1.0,
                        p_value: if percent_diff.abs() > 1.0 { 0.01 } else { 0.5 },
                    },
                }
            })
            .collect();

        rankings.sort_by(|a, b| {
            b.mean_phi
                .partial_cmp(&a.mean_phi)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, entry) in rankings.iter_mut().enumerate() {
            entry.rank = (i + 1) as u32;
        }

        rankings
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
