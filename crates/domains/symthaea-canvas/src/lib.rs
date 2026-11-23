// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-canvas
//!
//! Renderer-neutral living topology model with deterministic SVG and 3D mesh adapters.
//!
//! Converts cognitive telemetry (consciousness level, neuromodulators, Betti numbers,
//! Cantor depth, harmony activations) into a declarative scene graph, then renders
//! self-contained animated SVG or portable indexed 3D tube geometry.
//!
//! Pipeline: `CognitiveSnapshot → AestheticEngine → AestheticState → build_scene() → render_svg()`
//!
//! Total pipeline target: <500µs per frame.

#![deny(unsafe_code)]

pub mod aesthetic_engine;
pub mod aesthetic_score;
pub mod animation;
pub mod cognitive_events;
pub mod cognitive_layout;
pub mod cognitive_morphology;
pub mod cognitive_svg;
pub mod cognitive_topology;
pub mod color;
pub mod embodiment;
pub mod embodiment_trace;
pub mod geometry;
pub mod mycelial_acquisition_domain;
pub mod mycelial_acquisition_domain;
pub mod mycelial_annotation_triage;
pub mod mycelial_contact_persistence;
pub mod mycelial_failure_resilience;
pub mod mycelial_fusion_counterfactual;
pub mod mycelial_graph_hypothesis_posterior;
pub mod mycelial_junction_graph;
pub mod mycelial_mesh;
pub mod mycelial_model;
pub mod mycelial_network;
pub mod mycelial_observation_coverage;
pub mod mycelial_presence;
pub mod mycelial_reconstruction_calibration;
pub mod mycelial_reconstruction_metrics;
pub mod mycelial_reference_networks;
pub mod mycelial_temporal_cycles;
pub mod scene_features;
pub mod scene_graph;
pub mod svg_renderer;
pub mod validation;

use serde::{Deserialize, Serialize};

// Re-exports
pub use aesthetic_engine::{AestheticEngine, AestheticState};
pub use aesthetic_score::aesthetic_score;
pub use animation::{FrameContext, MotionPreference};
pub use cognitive_events::{
    COGNITIVE_TOPOLOGY_EVENT_SCHEMA_VERSION, CognitiveEventApplyOutcome, CognitiveEventError,
    CognitiveItemDelta, CognitiveTopologyChange, CognitiveTopologyDiff, CognitiveTopologyEvent,
    CognitiveTopologyReducer, MAX_COGNITIVE_EVENT_LOG, MAX_COGNITIVE_EVENTS_PER_BATCH,
    diff_cognitive_topologies,
};
pub use cognitive_layout::{
    COGNITIVE_PROJECTION_SCHEMA_VERSION, CognitiveAnchor, CognitiveLayoutConfig,
    CognitiveLayoutError, CognitiveSpatialLink, CognitiveSpatialProjection,
    layout_cognitive_topology, project_cognitive_topology,
};
pub use cognitive_morphology::{
    COGNITIVE_MORPHOLOGY_SCHEMA_VERSION, CognitiveHypha, CognitiveMorphologyConfig,
    CognitiveMorphologyError, CognitiveMorphologyNode, CognitiveMorphologyProjection,
    build_cognitive_morphology,
};
pub use cognitive_svg::{
    CognitiveSvgError, CognitiveSvgOptions, render_cognitive_morphology_svg,
    render_cognitive_projection_svg,
};
pub use cognitive_topology::{
    COGNITIVE_TOPOLOGY_SCHEMA_VERSION, CognitiveEdge, CognitiveEpisode, CognitiveEpisodeKind,
    CognitiveEpisodeState, CognitiveEvidence, CognitiveNode, CognitiveNodeKind,
    CognitiveProvenance, CognitiveRelation, CognitiveTopology, CognitiveTopologyError,
    EpistemicStatus, MAX_COGNITIVE_EDGES, MAX_COGNITIVE_EPISODES, MAX_COGNITIVE_NODES,
};
pub use color::{Color, Palette};
pub use embodiment::{
    EmbodimentActivity, EmbodimentController, EmbodimentEvent, EmbodimentEventKind,
    EmbodimentEventSequencer, EmbodimentIngestError, EmbodimentSequenceError, EmbodimentSnapshot,
    GazeTarget, Observed, SignalProvenance, SpeechOutputState, Viseme, VisemeSignal,
};
pub use embodiment_trace::{
    EMBODIMENT_TRACE_SCHEMA_VERSION, EmbodimentSession, EmbodimentSessionError, EmbodimentTrace,
    EmbodimentTraceError, EmbodimentTraceSummary, MAX_EMBODIMENT_SESSION_ID_BYTES,
    MAX_EMBODIMENT_TRACE_EVENTS, OpenActivityInterval, OpenPlaybackInterval,
};
pub use geometry::build_scene;
pub use mycelial_acquisition_domain::{
    MAX_MYCELIAL_ACQUISITION_DOMAIN_UNITS, MYCELIAL_ACQUISITION_DOMAIN_SCHEMA_VERSION,
    MYCELIAL_EXTRACTION_RECEIPT_SCHEMA_VERSION, MycelialAcquisitionAxis,
    MycelialAcquisitionAxisName, MycelialAcquisitionDomainError, MycelialAcquisitionDomainManifest,
    MycelialAcquisitionDomainSpec, MycelialAcquisitionScopeExclusion, MycelialAcquisitionUnitKey,
    MycelialExtractionCoverageReceipt, MycelialExtractionCoverageStatus,
    MycelialExtractionCoverageSummary, MycelialExtractionUnitOutcome, MycelialExtractionUnitState,
    MycelialSourceRightsStatus,
};
pub use mycelial_annotation_triage::{
    MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS, MYCELIAL_ANNOTATION_TRIAGE_SCHEMA_VERSION,
    MycelialAnnotationTriageConfig, MycelialAnnotationTriageError, MycelialAnnotationTriageReport,
    MycelialAnnotationTriageTopologySummary, MycelialJunctionReviewItem,
    MycelialReviewOrderingPolicy, build_mycelial_annotation_triage,
};
pub use mycelial_contact_persistence::{
    DEFAULT_MAX_CONTACT_TOLERANCE_MODEL_UNITS, MAX_CONTACT_FILTRATION_CANDIDATES,
    MYCELIAL_CONTACT_FILTRATION_SCHEMA_VERSION, MycelialContactBaselineEdge,
    MycelialContactCandidateObservation, MycelialContactComponentMerge,
    MycelialContactConflictGroup, MycelialContactCycleBirth, MycelialContactFiltration,
    MycelialContactFiltrationConfig, MycelialContactFiltrationError,
    MycelialContactFiltrationMetrics, MycelialContactFiltrationSnapshot, MycelialCyclePathStep,
    build_mycelial_contact_filtration,
};
pub use mycelial_failure_resilience::{
    MAX_FAILURE_DELIVERY_EVALUATIONS, MAX_FAILURE_DELIVERY_TRIALS_PER_SCENARIO,
    MAX_FAILURE_DEMAND_PAIRS, MAX_FAILURE_DEMAND_SOURCES, MAX_FAILURE_ID_BYTES,
    MAX_FAILURE_JUNCTIONS, MAX_FAILURE_SCENARIOS, MAX_FAILURE_SEGMENTS,
    MAX_FAILURE_TOTAL_DEMAND_WEIGHT, MYCELIAL_FAILURE_ATTACK_PLAN_SCHEMA_VERSION,
    MYCELIAL_FAILURE_DELIVERY_PROFILE_SCHEMA_VERSION, MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION,
    MycelialDemandPair, MycelialDemandPairOutcome, MycelialDemandPathLoad,
    MycelialFailureAttackConfig, MycelialFailureAttackEvaluation, MycelialFailureAttackPlan,
    MycelialFailureAttackStrategy, MycelialFailureDeliveryConfig, MycelialFailureDeliveryProfile,
    MycelialFailureDeliveryScenarioSummary, MycelialFailureDeliveryTrial, MycelialFailureMetrics,
    MycelialFailureResilienceError, MycelialFailureResilienceProfile, MycelialFailureScenario,
    build_mycelial_failure_attack_plan, build_mycelial_failure_delivery_profile,
    build_mycelial_failure_resilience_profile, build_single_failure_resilience_profile,
    evaluate_mycelial_failure_attack_plan,
};
pub use mycelial_fusion_counterfactual::{
    MAX_MYCELIAL_FUSION_COUNTERFACTUAL_CANDIDATES, MAX_MYCELIAL_FUSION_COUNTERFACTUAL_JUNCTIONS,
    MAX_MYCELIAL_FUSION_COUNTERFACTUAL_SEGMENTS, MAX_MYCELIAL_FUSION_DEMAND_EVALUATIONS,
    MAX_MYCELIAL_FUSION_DEMAND_PAIRS, MAX_MYCELIAL_FUSION_DISRUPTION_SCENARIOS,
    MAX_MYCELIAL_FUSION_GRAPH_VIEW_EVALUATIONS, MAX_MYCELIAL_FUSION_ID_BYTES,
    MAX_MYCELIAL_FUSION_TOTAL_DEMAND_WEIGHT, MAX_SIDE_CONTACT_FRACTION_MICROS,
    MIN_SIDE_CONTACT_FRACTION_MICROS, MYCELIAL_FUSION_COUNTERFACTUAL_SCHEMA_VERSION,
    MycelialFusionBridgeLengthBasis, MycelialFusionCandidate, MycelialFusionCandidateImpact,
    MycelialFusionCounterfactualError, MycelialFusionCounterfactualReport,
    MycelialFusionDemandDelta, MycelialFusionDemandOutcome, MycelialFusionDemandPair,
    MycelialFusionDisruptionDelta, MycelialFusionDisruptionScenario, MycelialFusionTarget,
    MycelialFusionTopologyEffect, MycelialFusionViewMetrics,
    evaluate_mycelial_fusion_counterfactuals,
};
pub use mycelial_graph_hypothesis_posterior::{
    MAX_MYCELIAL_POSTERIOR_CANDIDATES, MAX_MYCELIAL_POSTERIOR_HYPOTHESES,
    MYCELIAL_GRAPH_HYPOTHESIS_POSTERIOR_SCHEMA_VERSION, MycelialFusionCandidateMarginal,
    MycelialFusionConflictPair, MycelialGraphHypothesis, MycelialGraphHypothesisPosterior,
    MycelialGraphHypothesisPosteriorError, MycelialGraphHypothesisPrior,
    evaluate_mycelial_graph_hypothesis_posterior,
};

pub use mycelial_junction_graph::{
    GeometricEfficiencyReport, JunctionKind, JunctionLengthUnit, MAX_JUNCTION_GRAPH_JUNCTIONS,
    MAX_JUNCTION_GRAPH_SEGMENTS, MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION, MycelialHyphaSegment,
    MycelialJunction, MycelialJunctionGraphConfig, MycelialJunctionGraphError,
    MycelialJunctionGraphMetrics, MycelialJunctionHyphaGraph, PARAMETER_SCALE,
    build_mycelial_junction_hypha_graph,
};
pub use mycelial_reconstruction_calibration::{
    JunctionCalibrationObservation, JunctionClassProbabilityPrediction,
    MAX_RECONSTRUCTION_CALIBRATION_BINS, MAX_RECONSTRUCTION_CALIBRATION_PREDICTIONS,
    MYCELIAL_RECONSTRUCTION_CALIBRATION_SCHEMA_VERSION, MycelialReconstructionCalibrationError,
    MycelialReconstructionCalibrationReport, ReconstructionCalibrationBin,
    ReconstructionCalibrationClass, ReconstructionCalibrationConfig,
    ReconstructionCalibrationCounts, ReconstructionClassCalibration,
    ReconstructionSelectiveRiskPoint, ReconstructionSelectiveRiskSummary,
    ReconstructionTopLabelCalibration, evaluate_mycelial_reconstruction_calibration,
};
pub use mycelial_reconstruction_metrics::{
    AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind, AnnotatedMycelialSkeleton,
    JunctionIdentityMatch, JunctionKindScore, MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH,
    MAX_RECONSTRUCTION_ID_BYTES, MAX_RECONSTRUCTION_JUNCTIONS, MAX_RECONSTRUCTION_SEGMENTS,
    MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, MYCELIAL_RECONSTRUCTION_SCORE_SCHEMA_VERSION,
    MycelialReconstructionError, MycelialReconstructionScore, ReconstructionComponentAttribution,
    ReconstructionComponentErrorSummary, ReconstructionCounts, ReconstructionDimensionality,
    ReconstructionEvaluationConfig, ReconstructionLengthError, ReconstructionLengthUnit,
    ReconstructionLocalizationError, ReconstructionPoint3, ReconstructionTopologyDelta,
    ReconstructionTopologySummary, evaluate_mycelial_reconstruction, match_junctions_hungarian,
};
pub use mycelial_reference_networks::{
    MAX_REFERENCE_NETWORK_JUNCTIONS, MYCELIAL_EUCLIDEAN_MST_SCHEMA_VERSION,
    MycelialEuclideanMstEdge, MycelialEuclideanMstReport, MycelialMinimumSpanningForestReport,
    MycelialReferenceNetworkError, SpatialReferenceDimensionality,
    SpatialReferenceEfficiencyReport, build_euclidean_minimum_spanning_tree,
    build_observed_network_minimum_spanning_forest,
};
pub use mycelial_temporal_cycles::{
    MAX_TEMPORAL_CYCLE_RECORDS, MAX_TEMPORAL_CYCLE_SEARCH_STEPS,
    MAX_TEMPORAL_CYCLE_WINDOW_OBSERVATIONS, MAX_TEMPORAL_CYCLES_PER_WINDOW,
    MAX_TEMPORAL_EDGE_WINDOW_OBSERVATIONS, MAX_TEMPORAL_MYCELIAL_EDGES,
    MAX_TEMPORAL_MYCELIAL_NODES, MAX_TEMPORAL_MYCELIAL_WINDOWS,
    MYCELIAL_TEMPORAL_CYCLE_PROFILE_SCHEMA_VERSION, MYCELIAL_TEMPORAL_CYCLE_SCHEMA_VERSION,
    MycelialTemporalCycleError, TemporalCycleWindowObservation, TemporalCycleWindowState,
    TemporalEdgeEvidence, TemporalMycelialCycleConfig, TemporalMycelialCycleHyperedge,
    TemporalMycelialCycleProfile, TemporalMycelialCycleRecord, TemporalMycelialEdge,
    TemporalMycelialEdgeEvidenceCounts, TemporalMycelialEdgeEvidenceRecord,
    TemporalMycelialNodeMetrics, TemporalMycelialSequence, TemporalMycelialSnapshot,
    TemporalMycelialSnapshotMetrics, TemporalWindowCoverage, build_temporal_mycelial_cycle_profile,
    temporal_snapshot_from_annotated_skeleton,
};

pub use mycelial_mesh::{
    FilamentMeshRange, MAX_MYCELIAL_FILAMENTS, MAX_TUBE_MESH_VERTICES, MycelialTubeMesh,
    TubeTessellationConfig, TubeTessellationError, tessellate_mycelial_model,
};
pub use mycelial_model::{
    MYCELIAL_MODEL_VERSION, MycelialFilament, MycelialModel, OrbitalCell, PresenceMode,
    SpatialPoint3, build_mycelial_model,
};
pub use mycelial_network::{
    AnastomosisKind, MAX_MYCELIAL_ANASTOMOSES, MYCELIAL_NETWORK_SCHEMA_VERSION,
    MycelialAnastomosis, MycelialAnastomosisCandidate, MycelialConnectionKind,
    MycelialNetworkConfig, MycelialNetworkConnection, MycelialNetworkError, MycelialNetworkGraph,
    MycelialNetworkMetrics, build_mycelial_network_graph, detect_anastomosis_candidates,
};
pub use mycelial_observation_coverage::{
    MAX_MYCELIAL_COVERAGE_CANDIDATES_PER_OBSERVATION, MAX_MYCELIAL_OBSERVATION_COVERAGE_RECORDS,
    MYCELIAL_OBSERVATION_COVERAGE_SCHEMA_VERSION, MycelialObservationCoverageError,
    MycelialObservationCoverageLedger, MycelialObservationCoverageRecord,
    MycelialObservationCoverageState, MycelialObservationResidualKind,
    mycelial_observation_id_inventory_digest,
};
pub use mycelial_presence::{
    build_mycelial_presence, build_mycelial_presence_with_mode, render_mycelial_presence,
    render_mycelial_presence_with_mode,
};
pub use scene_features::{SceneFeatures, extract_scene_features};
pub use scene_graph::SceneNode;
pub use svg_renderer::{SvgRenderOptions, render_svg, render_svg_with_options};
pub use validation::{SnapshotLimits, SnapshotSanitization};

/// Lightweight snapshot of cognitive state, decoupled from CycleMetadata.
///
/// Extracted once per generation interval from the cognitive loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSnapshot {
    /// Consciousness level Ψ (0.0–1.0).
    pub consciousness_level: f64,
    /// Current prediction error magnitude.
    pub prediction_error: f32,
    /// Living mind vitality measure.
    pub living_mind_vitality: f64,
    /// Living mind coherence measure.
    pub living_mind_coherence: f64,

    // Neuromodulators (effective levels, 0.0–1.0)
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub oxytocin: f32,
    pub gaba: f32,
    pub allostatic_load: f32,

    // Topology
    /// Betti-0: connected components.
    pub betti_0: usize,
    /// Betti-1: loops/cycles.
    pub betti_1: usize,
    /// Betti-2: voids/cavities.
    pub betti_2: usize,
    /// Persistence diagram: birth-death pairs for components.
    pub persistence_components: Vec<[f64; 2]>,
    /// Persistence diagram: birth-death pairs for cycles.
    pub persistence_cycles: Vec<[f64; 2]>,

    // Cantor fractal
    pub cantor_metacognitive_depth: f32,
    pub cantor_last_depth: u8,

    // Affect
    /// Valence in [-1, 1].
    pub valence: f32,
    /// Arousal in [0, 1].
    pub arousal: f32,

    // Harmonies
    /// Eight Harmony activations (0.0–1.0 each).
    pub harmony_activations: [f32; 8],

    // Spatial
    /// Low-dimensional thought vector (typically 32D, first 2 dims used for layout).
    pub thought_vector: Vec<f32>,
    /// Monotonic cycle counter.
    pub cycle_count: u64,
}

impl CognitiveSnapshot {
    /// Create a default/dormant snapshot (minimal consciousness).
    pub fn dormant() -> Self {
        Self {
            consciousness_level: 0.05,
            prediction_error: 0.0,
            living_mind_vitality: 0.0,
            living_mind_coherence: 0.0,
            dopamine: 0.1,
            noradrenaline: 0.1,
            serotonin: 0.3,
            acetylcholine: 0.1,
            oxytocin: 0.2,
            gaba: 0.5,
            allostatic_load: 0.0,
            betti_0: 1,
            betti_1: 0,
            betti_2: 0,
            persistence_components: Vec::new(),
            persistence_cycles: Vec::new(),
            cantor_metacognitive_depth: 0.0,
            cantor_last_depth: 0,
            valence: 0.0,
            arousal: 0.1,
            harmony_activations: [0.0; 8],
            thought_vector: vec![0.0, 0.0],
            cycle_count: 0,
        }
    }
}

/// Full pipeline: snapshot → SVG string.
///
/// Convenience function that runs the entire canvas pipeline.
/// For stateful usage (EMA smoothing), use `AestheticEngine` directly.
pub fn render_snapshot(snap: &CognitiveSnapshot) -> (String, AestheticState) {
    let frame = FrameContext::from_cycle_count(snap.cycle_count);
    render_snapshot_frame(snap, frame)
}

/// Full pipeline on an explicit monotonic frame timeline.
pub fn render_snapshot_frame(
    snap: &CognitiveSnapshot,
    frame: FrameContext,
) -> (String, AestheticState) {
    let mut engine = AestheticEngine::new();
    let mut state = engine.process_frame(snap, frame);
    let scene = build_scene(&state);
    state.aesthetic_score = aesthetic_score(&state, &scene);
    let svg = render_svg_with_options(
        &scene,
        snap.consciousness_level,
        SvgRenderOptions::from_frame(frame),
    );
    (svg, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dormant_snapshot_renders() {
        let snap = CognitiveSnapshot::dormant();
        let (svg, state) = render_snapshot(&snap);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(state.luminosity < 0.2);
    }

    #[test]
    fn full_pipeline_deterministic() {
        let snap = CognitiveSnapshot {
            consciousness_level: 0.7,
            prediction_error: 0.15,
            living_mind_vitality: 0.6,
            living_mind_coherence: 0.5,
            dopamine: 0.6,
            noradrenaline: 0.4,
            serotonin: 0.5,
            acetylcholine: 0.3,
            oxytocin: 0.4,
            gaba: 0.5,
            allostatic_load: 0.2,
            betti_0: 4,
            betti_1: 2,
            betti_2: 1,
            persistence_components: vec![[0.0, 0.5], [0.1, 0.8]],
            persistence_cycles: vec![[0.2, 0.6]],
            cantor_metacognitive_depth: 0.7,
            cantor_last_depth: 3,
            valence: 0.3,
            arousal: 0.6,
            harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
            thought_vector: vec![0.3, -0.2],
            cycle_count: 500,
        };
        let (svg1, state1) = render_snapshot(&snap);
        let (svg2, state2) = render_snapshot(&snap);
        assert_eq!(svg1, svg2, "SVG should be deterministic");
        assert_eq!(state1.aesthetic_score, state2.aesthetic_score);
    }

    #[test]
    fn low_vs_high_consciousness_differ() {
        let low = CognitiveSnapshot::dormant();
        let mut high = CognitiveSnapshot::dormant();
        high.consciousness_level = 0.9;
        high.betti_0 = 5;
        high.betti_1 = 3;
        high.betti_2 = 1;
        high.arousal = 0.7;
        high.harmony_activations = [0.8; 8];
        high.cantor_last_depth = 4;

        let (svg_low, state_low) = render_snapshot(&low);
        let (svg_high, state_high) = render_snapshot(&high);

        assert!(state_high.luminosity > state_low.luminosity);
        assert!(
            svg_high.len() > svg_low.len(),
            "high Ψ should produce more SVG"
        );
    }

    #[test]
    fn aesthetic_score_bounded() {
        let snap = CognitiveSnapshot {
            consciousness_level: 0.6,
            prediction_error: 0.1,
            living_mind_vitality: 0.5,
            living_mind_coherence: 0.5,
            dopamine: 0.5,
            noradrenaline: 0.3,
            serotonin: 0.5,
            acetylcholine: 0.4,
            oxytocin: 0.3,
            gaba: 0.5,
            allostatic_load: 0.2,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            persistence_components: vec![],
            persistence_cycles: vec![],
            cantor_metacognitive_depth: 0.5,
            cantor_last_depth: 2,
            valence: 0.0,
            arousal: 0.4,
            harmony_activations: [0.5; 8],
            thought_vector: vec![0.0, 0.0],
            cycle_count: 200,
        };
        let (_, state) = render_snapshot(&snap);
        assert!(state.aesthetic_score >= 0.0 && state.aesthetic_score <= 1.0);
    }

    #[test]
    fn snapshot_serialization() {
        let snap = CognitiveSnapshot::dormant();
        let json = serde_json::to_string(&snap).expect("serialize");
        let back: CognitiveSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(snap.consciousness_level, back.consciousness_level);
        assert_eq!(snap.betti_0, back.betti_0);
    }
}
