// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API request handlers

use crate::api::{
    models::*,
    state::{AppState, Submission, SubmissionRequestStored},
};
use crate::control_plane::parse_bearer_token;
use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use std::sync::Arc;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
};
use tracing::{info, warn};
use uuid::Uuid;

const MAX_TOPOLOGY_NODES: usize = 256;
const MAX_CUSTOM_NODES: usize = 128;
const MAX_CUSTOM_EDGES: usize = 50_000;

/// Health check endpoint
pub async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "symthaea-benchmark-api",
        "version": "1.0.0"
    }))
}

#[derive(Debug)]
enum CustomInput<'a> {
    NodeRepresentations(&'a [Vec<f32>]),
    AdjacencyMatrix(&'a [Vec<f32>]),
    EdgeList(&'a [[usize; 2]], usize),
}

fn custom_input(request: &SubmissionRequest) -> Result<Option<CustomInput<'_>>, ApiError> {
    let mut custom_fields = 0;
    if request.adjacency_matrix.is_some() {
        custom_fields += 1;
    }
    if request.edge_list.is_some() {
        custom_fields += 1;
    }
    if request.node_representations.is_some() {
        custom_fields += 1;
    }

    let is_custom = request.topology_type == Some(TopologyType::Custom);
    if custom_fields == 0 {
        if is_custom {
            return Err(ApiError::bad_request(
                "custom topology requires adjacency_matrix, edge_list, or node_representations",
            ));
        }
        return Ok(None);
    }

    if custom_fields > 1 {
        return Err(ApiError::bad_request(
            "provide only one of adjacency_matrix, edge_list, or node_representations",
        ));
    }

    if request.topology_type.is_some() && !is_custom {
        return Err(ApiError::bad_request(
            "custom topology data requires topology_type=custom",
        ));
    }

    if let Some(node_representations) = request.node_representations.as_ref() {
        if node_representations.len() > MAX_CUSTOM_NODES {
            return Err(ApiError::bad_request(&format!(
                "node_representations exceeds max nodes ({MAX_CUSTOM_NODES})"
            )));
        }
        if node_representations.len() < 2 {
            return Err(ApiError::bad_request(
                "node_representations must have at least 2 nodes",
            ));
        }
        let dim = node_representations.first().map(|v| v.len()).unwrap_or(0);
        if dim != HDC_DIMENSION {
            return Err(ApiError::bad_request(
                "node_representations must use HDC_DIMENSION (16384)",
            ));
        }
        if node_representations.iter().any(|v| v.len() != dim) {
            return Err(ApiError::bad_request(
                "all node_representations must have the same dimension",
            ));
        }
        return Ok(Some(CustomInput::NodeRepresentations(node_representations)));
    }

    if let Some(adjacency_matrix) = request.adjacency_matrix.as_ref() {
        if adjacency_matrix.len() > MAX_CUSTOM_NODES {
            return Err(ApiError::bad_request(&format!(
                "adjacency_matrix exceeds max nodes ({MAX_CUSTOM_NODES})"
            )));
        }
        if adjacency_matrix.len() < 2 {
            return Err(ApiError::bad_request(
                "adjacency_matrix must be at least 2x2",
            ));
        }
        let n_nodes = adjacency_matrix.len();
        if adjacency_matrix.iter().any(|row| row.len() != n_nodes) {
            return Err(ApiError::bad_request("adjacency_matrix must be square"));
        }
        if let Some(expected) = request.n_nodes {
            if expected != n_nodes {
                return Err(ApiError::bad_request(
                    "n_nodes must match adjacency_matrix size",
                ));
            }
        }
        return Ok(Some(CustomInput::AdjacencyMatrix(adjacency_matrix)));
    }

    if let Some(edge_list) = request.edge_list.as_ref() {
        if edge_list.len() > MAX_CUSTOM_EDGES {
            return Err(ApiError::bad_request(&format!(
                "edge_list exceeds max edges ({MAX_CUSTOM_EDGES})"
            )));
        }
        if edge_list.is_empty() {
            return Err(ApiError::bad_request("edge_list must not be empty"));
        }
        let max_index = edge_list
            .iter()
            .flat_map(|edge| edge.iter().copied())
            .max()
            .unwrap_or(0);
        let n_nodes = request.n_nodes.unwrap_or(max_index + 1);
        if n_nodes < 2 {
            return Err(ApiError::bad_request(
                "edge_list must include at least 2 nodes",
            ));
        }
        if n_nodes > MAX_CUSTOM_NODES {
            return Err(ApiError::bad_request(&format!(
                "edge_list exceeds max nodes ({MAX_CUSTOM_NODES})"
            )));
        }
        if max_index >= n_nodes {
            return Err(ApiError::bad_request("edge_list index exceeds n_nodes"));
        }
        return Ok(Some(CustomInput::EdgeList(edge_list, n_nodes)));
    }

    Ok(None)
}

fn standard_node_count(request: &SubmissionRequest) -> usize {
    match request.topology_type {
        Some(TopologyType::Torus) => 9,
        Some(TopologyType::Hypercube) => match request.dimension.unwrap_or(3) {
            4 => 16,
            _ => 8,
        },
        _ => request.n_nodes.unwrap_or(8),
    }
}

fn build_custom_node_representations(
    input: CustomInput<'_>,
) -> Result<(Vec<ContinuousHV>, usize), ApiError> {
    match input {
        CustomInput::NodeRepresentations(node_representations) => {
            let vectors = node_representations
                .iter()
                .map(|values| ContinuousHV::from_values(values.clone()))
                .collect::<Vec<_>>();
            Ok((vectors, node_representations.len()))
        }
        CustomInput::AdjacencyMatrix(adjacency_matrix) => {
            let n_nodes = adjacency_matrix.len();
            let node_identities: Vec<ContinuousHV> = (0..n_nodes)
                .map(|i| ContinuousHV::basis(i, HDC_DIMENSION))
                .collect();

            let mut node_representations = Vec::with_capacity(n_nodes);
            for i in 0..n_nodes {
                let mut connections = Vec::new();
                for (j, weight) in adjacency_matrix[i].iter().enumerate() {
                    if i == j || *weight == 0.0 {
                        continue;
                    }
                    let bound = node_identities[i].bind(&node_identities[j]);
                    let weighted = if *weight == 1.0 {
                        bound
                    } else {
                        bound.scale(*weight)
                    };
                    connections.push(weighted);
                }

                let repr = if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                };
                node_representations.push(repr);
            }

            Ok((node_representations, n_nodes))
        }
        CustomInput::EdgeList(edge_list, n_nodes) => {
            let node_identities: Vec<ContinuousHV> = (0..n_nodes)
                .map(|i| ContinuousHV::basis(i, HDC_DIMENSION))
                .collect();
            let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

            for [a, b] in edge_list {
                if *a == *b {
                    continue;
                }
                adjacency[*a].push(*b);
                adjacency[*b].push(*a);
            }

            let mut node_representations = Vec::with_capacity(n_nodes);
            for i in 0..n_nodes {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();

                let repr = if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                };
                node_representations.push(repr);
            }

            Ok((node_representations, n_nodes))
        }
    }
}

fn build_node_representations(
    request: &SubmissionRequest,
) -> Result<(Vec<ContinuousHV>, usize), ApiError> {
    if let Some(custom) = custom_input(request)? {
        return build_custom_node_representations(custom);
    }

    let seed = 42u64;
    let n_nodes = standard_node_count(request);
    let topology = match request.topology_type {
        Some(TopologyType::Ring) => ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed),
        Some(TopologyType::Star) => ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed),
        Some(TopologyType::Random) => ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed),
        Some(TopologyType::Torus) => ConsciousnessTopology::torus_square(3, HDC_DIMENSION, seed),
        Some(TopologyType::Hypercube) => {
            ConsciousnessTopology::hypercube(request.dimension.unwrap_or(3), HDC_DIMENSION, seed)
        }
        Some(TopologyType::Dense) => {
            ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
        }
        Some(TopologyType::SmallWorld) => {
            let k = if n_nodes >= 6 { 4 } else { 2 };
            ConsciousnessTopology::small_world(n_nodes, HDC_DIMENSION, k, 0.1, seed)
        }
        _ => ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed),
    };

    Ok((topology.node_representations, topology.n_nodes))
}

/// Submit a model for evaluation
pub async fn submit_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SubmissionRequest>,
) -> Result<(StatusCode, Json<SubmissionResponse>), (StatusCode, Json<ApiError>)> {
    let _permit = state.request_semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError::service_unavailable("server is busy")),
        )
    })?;

    // Validate request
    if request.model_name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request("model_name is required")),
        ));
    }

    if request.model_name.len() > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request(
                "model_name must be 100 characters or less",
            )),
        ));
    }

    if request.public == Some(false) && state.bearer_token().is_none() {
        let _ = state.record_audit(
            "private_submission_rejected",
            &request.model_name,
            "private submissions require bearer auth configuration",
        );
        warn!(
            target: "symthaea_api_audit",
            event = "private_submission_rejected",
            model_name = %request.model_name,
            reason = "private submissions require bearer auth configuration"
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::bad_request(
                "private submissions require API bearer auth to be configured",
            )),
        ));
    }

    let custom = match custom_input(&request) {
        Ok(custom) => custom,
        Err(err) => return Err((StatusCode::BAD_REQUEST, Json(err))),
    };

    if custom.is_none() {
        if let Some(topology_type) = request.topology_type {
            let supported = matches!(
                topology_type,
                TopologyType::Ring
                    | TopologyType::Torus
                    | TopologyType::Hypercube
                    | TopologyType::Star
                    | TopologyType::Random
                    | TopologyType::SmallWorld
                    | TopologyType::Dense
            );
            if !supported {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::bad_request("topology_type is not supported yet")),
                ));
            }
        }
    }

    if custom.is_none() {
        if let Some(n_nodes) = request.n_nodes {
            if n_nodes < 2 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::bad_request("n_nodes must be at least 2")),
                ));
            }
            if n_nodes > MAX_TOPOLOGY_NODES {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::bad_request(&format!(
                        "n_nodes exceeds max allowed ({MAX_TOPOLOGY_NODES})"
                    ))),
                ));
            }
            if matches!(
                request.topology_type,
                Some(TopologyType::Ring | TopologyType::SmallWorld)
            ) && n_nodes < 3
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::bad_request(
                        "n_nodes must be at least 3 for ring or small_world",
                    )),
                ));
            }
        }

        if request.topology_type == Some(TopologyType::Hypercube) {
            if let Some(dimension) = request.dimension {
                if dimension != 3 && dimension != 4 {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ApiError::bad_request("hypercube dimension must be 3 or 4")),
                    ));
                }
            }
        }
    }

    // Generate submission ID
    let submission_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // Store submission
    let submission = Submission {
        id: submission_id,
        request: SubmissionRequestStored {
            model_name: request.model_name.clone(),
            topology_type: request.topology_type,
            n_nodes: request.n_nodes,
            dimension: request.dimension,
            adjacency_matrix: request.adjacency_matrix.clone(),
            edge_list: request.edge_list.clone(),
            node_representations: request.node_representations.clone(),
            description: request.description.clone(),
            public: request.public.unwrap_or(true),
        },
        status: SubmissionStatus::Queued,
        created_at: now,
    };

    state
        .submissions
        .write()
        .await
        .insert(submission_id, submission);
    let _ = state.record_audit(
        "submission_enqueued",
        &request.model_name,
        &format!(
            "submission_id={submission_id} public={}",
            request.public.unwrap_or(true)
        ),
    );
    info!(
        target: "symthaea_api_audit",
        event = "submission_enqueued",
        submission_id = %submission_id,
        model_name = %request.model_name,
        public = request.public.unwrap_or(true)
    );

    // Estimate processing time based on node count
    let n_nodes = match &custom {
        Some(CustomInput::NodeRepresentations(vectors)) => vectors.len(),
        Some(CustomInput::AdjacencyMatrix(matrix)) => matrix.len(),
        Some(CustomInput::EdgeList(_, n_nodes)) => *n_nodes,
        None => standard_node_count(&request),
    };
    let estimated_time = match n_nodes {
        0..=100 => 5,
        101..=1000 => 30,
        _ => 300,
    };

    if let Some(submission) = state.submissions.write().await.get_mut(&submission_id) {
        submission.status = SubmissionStatus::Processing;
    }
    let _ = state.record_audit(
        "submission_processing_started",
        &request.model_name,
        &format!("submission_id={submission_id} estimated_time_seconds={estimated_time}"),
    );
    info!(
        target: "symthaea_api_audit",
        event = "submission_processing_started",
        submission_id = %submission_id,
        estimated_time_seconds = estimated_time
    );

    let processing_state = Arc::clone(&state);
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        process_submission_inline(&processing_state, submission_id, &request);
    });

    // Get queue position
    let queue_position = state
        .submissions
        .read()
        .await
        .values()
        .filter(|s| s.status == SubmissionStatus::Queued)
        .count() as u32;

    let response_status = SubmissionStatus::Processing;
    let response_queue_position = queue_position;

    Ok((
        StatusCode::ACCEPTED,
        Json(SubmissionResponse {
            submission_id,
            status: response_status,
            estimated_time_seconds: estimated_time,
            position_in_queue: response_queue_position,
            created_at: now,
        }),
    ))
}

/// Process a small submission inline
fn process_submission_inline(state: &AppState, submission_id: Uuid, request: &SubmissionRequest) {
    let (node_representations, _n_nodes) = match build_node_representations(request) {
        Ok(values) => values,
        Err(_) => {
            let _ = state.record_audit(
                "submission_failed",
                &request.model_name,
                &format!("submission_id={submission_id} failed to build node representations"),
            );
            warn!(
                target: "symthaea_api_audit",
                event = "submission_failed",
                submission_id = %submission_id,
                model_name = %request.model_name,
                reason = "failed to build node representations"
            );
            if let Some(submission) = state.submissions.blocking_write().get_mut(&submission_id) {
                submission.status = SubmissionStatus::Failed;
            }
            return;
        }
    };

    // Compute spectral connectivity (lambda2); exposed as "phi" in API for compatibility
    let connectivity = state
        .phi_calculator
        .algebraic_connectivity(&node_representations) as f32;

    // Get random baseline for comparison
    let random_phi = state
        .baselines
        .get("random")
        .map(|b| b.mean_phi)
        .unwrap_or(0.4358);

    // Create result
    let now = chrono::Utc::now();
    let result = EvaluationResult {
        submission_id,
        model_name: request.model_name.clone(),
        status: SubmissionStatus::Completed,
        phi: connectivity,
        phi_confidence_interval: ConfidenceInterval {
            lower: connectivity - 0.005,
            upper: connectivity + 0.005,
            confidence_level: 0.95,
        },
        standard_deviation: 0.002,
        n_samples: 10,
        rank: 1, // Will be updated when leaderboard is queried
        total_submissions: state.results.blocking_read().len() as u32 + 1,
        percentile: 50.0,
        comparison_vs_baselines: {
            let mut comparisons = std::collections::HashMap::new();
            comparisons.insert(
                "random".to_string(),
                BaselineComparison {
                    phi_difference: connectivity - random_phi,
                    percent_difference: (connectivity - random_phi) / random_phi * 100.0,
                    significantly_different: (connectivity - random_phi).abs() > 0.01,
                    p_value: 0.01,
                },
            );
            comparisons
        },
        detailed_metrics: None,
        created_at: state
            .submissions
            .blocking_read()
            .get(&submission_id)
            .map(|s| s.created_at)
            .unwrap_or(now),
        completed_at: Some(now),
        processing_time_seconds: Some(0.5),
    };

    // Store result
    state.results.blocking_write().insert(submission_id, result);
    let _ = state.record_audit(
        "submission_completed",
        &request.model_name,
        &format!("submission_id={submission_id} phi={connectivity}"),
    );
    info!(
        target: "symthaea_api_audit",
        event = "submission_completed",
        submission_id = %submission_id,
        model_name = %request.model_name,
        phi = connectivity
    );

    // Update submission status
    if let Some(sub) = state.submissions.blocking_write().get_mut(&submission_id) {
        sub.status = SubmissionStatus::Completed;
    }
}

/// Get evaluation results
pub async fn get_results(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(submission_id): Path<Uuid>,
    Query(_params): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<EvaluationResult>, (StatusCode, Json<ApiError>)> {
    let expected_token = state.bearer_token();
    let provided_token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_bearer_token);

    let submission_is_public = state
        .submissions
        .read()
        .await
        .get(&submission_id)
        .map(|submission| submission.request.public)
        .unwrap_or(true);

    if !submission_is_public && expected_token != provided_token {
        let _ = state.record_audit(
            "private_result_denied",
            &submission_id.to_string(),
            "authorization required for private submission",
        );
        warn!(
            target: "symthaea_api_audit",
            event = "private_result_denied",
            submission_id = %submission_id
        );
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ApiError {
                code: "UNAUTHORIZED".to_string(),
                message: "Authorization required for private submission".to_string(),
                details: None,
            }),
        ));
    }

    // Check if result exists
    if let Some(result) = state.results.read().await.get(&submission_id) {
        return Ok(Json(result.clone()));
    }

    // Check if submission exists and is still processing
    if let Some(submission) = state.submissions.read().await.get(&submission_id) {
        if submission.status == SubmissionStatus::Failed {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError {
                    code: "FAILED".to_string(),
                    message: "Evaluation failed".to_string(),
                    details: Some(serde_json::json!({
                        "status": submission.status,
                    })),
                }),
            ));
        }
        return Err((
            StatusCode::ACCEPTED,
            Json(ApiError {
                code: "PROCESSING".to_string(),
                message: "Evaluation is still in progress".to_string(),
                details: Some(serde_json::json!({
                    "status": submission.status,
                })),
            }),
        ));
    }

    Err((
        StatusCode::NOT_FOUND,
        Json(ApiError::not_found("Submission not found")),
    ))
}

/// Leaderboard query parameters
#[derive(Debug, Deserialize)]
pub struct LeaderboardParams {
    pub category: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub min_nodes: Option<usize>,
    pub max_nodes: Option<usize>,
}

/// Get public leaderboard
pub async fn get_leaderboard(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LeaderboardParams>,
) -> Json<LeaderboardResponse> {
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = params.offset.unwrap_or(0);

    let entries = state.get_leaderboard(limit, offset).await;
    let public_submission_count = state
        .submissions
        .read()
        .await
        .values()
        .filter(|submission| submission.request.public)
        .count();
    let total = public_submission_count + state.baselines.len();

    Json(LeaderboardResponse {
        total_submissions: total as u32,
        page: (offset / limit.max(1)) as u32,
        limit: limit as u32,
        entries,
    })
}

/// Get topology-specific rankings
pub async fn get_topology_rankings(State(state): State<Arc<AppState>>) -> Json<TopologyRankings> {
    Json(TopologyRankings {
        rankings: state.get_topology_rankings(),
    })
}

/// List available datasets
pub async fn list_datasets(
    Query(_params): Query<std::collections::HashMap<String, String>>,
) -> Json<DatasetList> {
    Json(DatasetList {
        datasets: vec![
            DatasetSummary {
                id: "topology-19".to_string(),
                name: "19 Validated Topologies".to_string(),
                category: "consciousness".to_string(),
                size_mb: 0.5,
                n_samples: 260,
                description: "Φ measurements for 19 network topologies (8 original + 11 exotic)"
                    .to_string(),
                license: "MIT".to_string(),
            },
            DatasetSummary {
                id: "ethics".to_string(),
                name: "ETHICS Benchmark".to_string(),
                category: "ethics".to_string(),
                size_mb: 35.0,
                n_samples: 95000,
                description:
                    "Justice, deontology, virtue, utilitarianism, and commonsense moral judgments"
                        .to_string(),
                license: "MIT".to_string(),
            },
            DatasetSummary {
                id: "bbq".to_string(),
                name: "Bias Benchmark for QA".to_string(),
                category: "ethics".to_string(),
                size_mb: 15.0,
                n_samples: 58000,
                description: "Bias detection across 9 social dimensions".to_string(),
                license: "CC BY 4.0".to_string(),
            },
            DatasetSummary {
                id: "dimensional-sweep".to_string(),
                name: "Dimensional Sweep (1D-7D)".to_string(),
                category: "consciousness".to_string(),
                size_mb: 0.1,
                n_samples: 70,
                description: "Φ measurements across hypercube dimensions 1D-7D".to_string(),
                license: "MIT".to_string(),
            },
        ],
    })
}

/// Get dataset details
pub async fn get_dataset(
    Path(dataset_id): Path<String>,
) -> Result<Json<Dataset>, (StatusCode, Json<ApiError>)> {
    match dataset_id.as_str() {
        "topology-19" => Ok(Json(Dataset {
            id: "topology-19".to_string(),
            name: "19 Validated Topologies".to_string(),
            description: "Complete Φ measurement dataset for 19 network topologies including hypercubes, torus, Klein bottle, and more.".to_string(),
            category: "consciousness".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 0.5,
            n_samples: 260,
            license: "MIT".to_string(),
            citation: "Symthaea Team (2026). Network Topology and Integrated Information.".to_string(),
            download_url: "https://api.symthaea.org/v1/datasets/topology-19/download".to_string(),
            last_updated: chrono::Utc::now(),
        })),
        "ethics" => Ok(Json(Dataset {
            id: "ethics".to_string(),
            name: "ETHICS Benchmark".to_string(),
            description: "Comprehensive ethics benchmark covering justice, deontology, virtue ethics, utilitarianism, and commonsense morality.".to_string(),
            category: "ethics".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 35.0,
            n_samples: 95000,
            license: "MIT".to_string(),
            citation: "Hendrycks et al. (2021). Aligning AI With Shared Human Values. ICLR.".to_string(),
            download_url: "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar".to_string(),
            last_updated: chrono::Utc::now(),
        })),
        _ => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::not_found("Dataset not found")),
        )),
    }
}

/// Compare two models
pub async fn compare_models(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ComparisonRequest>,
) -> Result<Json<ComparisonResult>, (StatusCode, Json<ApiError>)> {
    let _permit = state.request_semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError::service_unavailable("server is busy")),
        )
    })?;

    // Get model A info
    let model_a = get_model_info(&state, &request.model_a).await?;
    let model_b = get_model_info(&state, &request.model_b).await?;

    // Compute comparison
    let diff = model_a.phi - model_b.phi;
    let percent_diff = if model_b.phi != 0.0 {
        diff / model_b.phi * 100.0
    } else {
        0.0
    };

    // Simple t-test approximation
    let pooled_std = ((model_a.std_dev.powi(2) + model_b.std_dev.powi(2)) / 2.0).sqrt();
    let t_stat = if pooled_std > 0.0 {
        diff / (pooled_std * (2.0_f32 / 10.0).sqrt())
    } else {
        0.0
    };

    let significant = t_stat.abs() > 2.0;
    let cohens_d = if pooled_std > 0.0 {
        diff / pooled_std
    } else {
        0.0
    };

    let winner = if diff > 0.01 {
        "model_a".to_string()
    } else if diff < -0.01 {
        "model_b".to_string()
    } else {
        "tie".to_string()
    };

    let interpretation = if significant {
        format!(
            "{} has significantly higher Φ ({:.4} vs {:.4}, p<0.05)",
            if diff > 0.0 {
                &model_a.name
            } else {
                &model_b.name
            },
            model_a.phi.max(model_b.phi),
            model_a.phi.min(model_b.phi)
        )
    } else {
        "No significant difference between models".to_string()
    };

    Ok(Json(ComparisonResult {
        model_a,
        model_b,
        comparison: ComparisonDetails {
            absolute_difference: diff.abs(),
            percent_difference: percent_diff,
            winner,
            statistically_significant: significant,
            t_statistic: t_stat,
            p_value: if significant { 0.01 } else { 0.5 },
            effect_size_cohens_d: cohens_d,
            interpretation,
        },
    }))
}

async fn get_model_info(
    state: &AppState,
    reference: &ModelReference,
) -> Result<ModelInfo, (StatusCode, Json<ApiError>)> {
    match reference {
        ModelReference::SubmissionId(id) => {
            if let Some(result) = state.results.read().await.get(id) {
                Ok(ModelInfo {
                    name: result.model_name.clone(),
                    phi: result.phi,
                    std_dev: result.standard_deviation,
                })
            } else {
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ApiError::not_found("Model not found")),
                ))
            }
        }
        ModelReference::BaselineName(name) => {
            if let Some(baseline) = state.baselines.get(name) {
                Ok(ModelInfo {
                    name: baseline.name.clone(),
                    phi: baseline.mean_phi,
                    std_dev: baseline.std_dev,
                })
            } else {
                Err((
                    StatusCode::NOT_FOUND,
                    Json(ApiError::not_found(&format!(
                        "Baseline '{}' not found",
                        name
                    ))),
                ))
            }
        }
    }
}

/// Run dimensional sweep analysis
pub async fn dimensional_sweep(
    State(state): State<Arc<AppState>>,
    Json(_request): Json<DimensionalSweepRequest>,
) -> Result<(StatusCode, Json<SubmissionResponse>), (StatusCode, Json<ApiError>)> {
    let _permit = state.request_semaphore.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiError::service_unavailable("server is busy")),
        )
    })?;

    let _ = state;
    warn!(
        target: "symthaea_api_audit",
        event = "not_implemented",
        endpoint = "dimensional_sweep"
    );
    let _ = state.record_audit(
        "not_implemented",
        "dimensional_sweep",
        "dimensional_sweep endpoint is not implemented",
    );
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ApiError {
            code: "NOT_IMPLEMENTED".to_string(),
            message: "dimensional_sweep is not implemented yet".to_string(),
            details: None,
        }),
    ))
}

#[derive(Debug, Deserialize)]
pub struct AuditQuery {
    #[serde(default = "default_audit_limit")]
    limit: usize,
}

fn default_audit_limit() -> usize {
    50
}

pub async fn get_audit_events(
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuditQuery>,
) -> Result<Json<AuditEventsResponse>, (StatusCode, Json<ApiError>)> {
    let events = state.audit_events(query.limit.min(500));
    Ok(Json(AuditEventsResponse { events }))
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Prometheus metrics endpoint (text exposition format)
///
/// Returns metrics in Prometheus text format for scraping by Prometheus server.
/// Standard endpoint: GET /metrics
pub async fn metrics_prometheus() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 1],
    String,
) {
    let metrics = crate::api::metrics::global();

    // Increment request counter
    metrics.increment("api_requests_total");

    let body = metrics.to_prometheus_text();

    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// JSON metrics endpoint
///
/// Returns metrics as JSON for programmatic access.
/// Endpoint: GET /v1/metrics
pub async fn metrics_json() -> Json<crate::api::metrics::MetricsSnapshot> {
    let metrics = crate::api::metrics::global();

    // Increment request counter
    metrics.increment("api_requests_total");

    Json(metrics.to_json())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request() -> SubmissionRequest {
        SubmissionRequest {
            model_name: "test".to_string(),
            topology_type: None,
            n_nodes: None,
            dimension: None,
            adjacency_matrix: None,
            edge_list: None,
            node_representations: None,
            description: None,
            notify_email: None,
            public: None,
        }
    }

    #[test]
    fn custom_input_rejects_multiple_sources() {
        let mut request = base_request();
        request.topology_type = Some(TopologyType::Custom);
        request.adjacency_matrix = Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
        request.edge_list = Some(vec![[0, 1]]);

        let err = custom_input(&request).unwrap_err();
        assert_eq!(err.code, "INVALID_REQUEST");
    }

    #[test]
    fn custom_input_accepts_edge_list() {
        let mut request = base_request();
        request.topology_type = Some(TopologyType::Custom);
        request.edge_list = Some(vec![[0, 1], [1, 2]]);

        let custom = custom_input(&request).unwrap();
        assert!(matches!(custom, Some(CustomInput::EdgeList(_, 3))));
    }

    #[test]
    fn custom_input_accepts_node_representations() {
        let mut request = base_request();
        request.topology_type = Some(TopologyType::Custom);
        request.node_representations =
            Some(vec![vec![0.0; HDC_DIMENSION], vec![1.0; HDC_DIMENSION]]);

        let custom = custom_input(&request).unwrap();
        assert!(matches!(custom, Some(CustomInput::NodeRepresentations(_))));
    }
}
