//! # FL Aggregator
//!
//! High-performance Byzantine-resistant federated learning aggregator.
//!
//! This crate provides production-ready gradient aggregation with:
//! - Byzantine fault tolerance (Krum, MultiKrum, Median, TrimmedMean)
//! - Memory-bounded operation for large-scale deployments
//! - Async support for concurrent node handling
//! - Optional Python bindings via PyO3
//!
//! ## Quick Start
//!
//! ```rust
//! use fl_aggregator::{Aggregator, AggregatorConfig, Defense};
//! use ndarray::Array1;
//!
//! // Create aggregator with Krum defense (tolerates 1 Byzantine node)
//! let config = AggregatorConfig::default()
//!     .with_defense(Defense::Krum { f: 1 })
//!     .with_expected_nodes(5);
//!
//! let mut aggregator = Aggregator::new(config);
//!
//! // Submit gradients from nodes
//! aggregator.submit("node1", Array1::from(vec![1.0, 2.0, 3.0])).unwrap();
//! aggregator.submit("node2", Array1::from(vec![1.1, 2.1, 3.1])).unwrap();
//! // ... more nodes
//!
//! // Get aggregated result
//! if aggregator.is_round_complete() {
//!     let result = aggregator.finalize_round().unwrap();
//! }
//! ```

pub mod adaptive;
pub mod aggregator;
pub mod attacks;
pub mod byzantine;
pub mod demo;
pub mod compression;
pub mod detection;
pub mod ensemble;
pub mod error;
pub mod hdc_byzantine;
pub mod hyperfeel;
pub mod metrics;
pub mod payload;
pub mod phi;
pub mod phi_series;
pub mod pogq;
pub mod shapley;
pub mod streaming;
pub mod replay;
pub mod unified_aggregator;
pub mod coordinator;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "http-api")]
pub mod http;

#[cfg(feature = "holochain")]
pub mod holochain;

#[cfg(feature = "privacy")]
pub mod privacy;

#[cfg(any(feature = "storage-local", feature = "storage-postgres"))]
pub mod storage;

#[cfg(feature = "identity")]
pub mod identity;

#[cfg(feature = "governance")]
pub mod governance;

#[cfg(feature = "proofs")]
pub mod proofs;

#[cfg(feature = "grpc")]
pub mod grpc;

#[cfg(feature = "prometheus")]
pub mod prometheus;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "ethereum")]
pub mod ethereum;

// Re-exports for convenience
pub use aggregator::{Aggregator, AggregatorConfig, AsyncAggregator};
#[cfg(feature = "proofs")]
pub use aggregator::verified::{VerifiedAggregator, VerifiedAsyncAggregator, VerifiedAggregatorStatus};
pub use byzantine::{Defense, DefenseConfig};
pub use compression::{CompressionConfig, GradientCompressor};
pub use detection::{ByzantineDetector, Classification, DetectionResult, DetectorConfig};
pub use error::{AggregatorError, Result};
pub use metrics::AggregatorMetrics;

// Payload types for unified FL
pub use payload::{
    Payload, PayloadType, UnifiedPayload,
    DenseGradient, Hypervector, BinaryHypervector,
    SparseGradient, QuantizedGradient,
    DenseAggregatable, HdcAggregatable,
    HYPERVECTOR_DIM,
};

// HyperFeel encoding/decoding
pub use hyperfeel::{
    encode_gradient, decode_gradient, EncodingConfig,
    calculate_compression_ratio, estimate_encoding_quality,
    calculate_cosine_similarity, EncodingQualityReport,
};

// HDC Byzantine-resistant aggregation
pub use hdc_byzantine::{
    HdcDefense, HdcDefenseConfig, HdcByzantineAggregator,
    HdcByzantineAggregate,
    component_median, component_median_binary,
    weighted_bundle, weighted_bundle_binary, softmax,
};

// Unified aggregator for multi-paradigm FL
pub use unified_aggregator::{
    UnifiedAggregator, UnifiedAggregatorConfig, AsyncUnifiedAggregator,
    UnifiedAggregationResult, UnifiedAggregatorStatus,
    PayloadSubmission,
};

// Adaptive defense system
pub use adaptive::{
    AdaptiveDefenseConfig, AdaptiveDefenseState, AdaptiveDefenseManager,
    AdaptiveStats, AdaptiveDefenseIntegration,
};

// Attack simulator for testing Byzantine defenses
pub use attacks::{
    AttackType, AttackSimulator, AttackScenario,
    DefenseTester, DefenseTestResult,
    ComparisonReport, ScenarioComparison, ComparisonSummary,
    AttackMetrics,
};

// Defense Ensemble (vote-based multi-defense aggregation)
pub use ensemble::{
    EnsembleConfig, EnsembleAggregator, EnsembleResult,
    VotingStrategy, ByzantineConsensus, AggregationFeedback,
    AgreementMetrics,
};

// Phi (Integrated Information) measurement and Byzantine detection
pub use phi::{
    PhiMeasurer, PhiConfig, PhiMetrics, PhiMeasurementResult,
    PhiByzantineResult, measure_phi, PHI_DIMENSION,
};

// Phi Time-Series tracking for consciousness/coherence evolution
pub use phi_series::{
    PhiTimeSeries, PhiTimeSeriesConfig, PhiDataPoint,
    PhiTrend, PhiAnomaly, PhiStatistics, AnomalySeverity,
    PhiStreamMonitor, DEFAULT_PHI_DIMENSION,
};

// PoGQ-v4.1 Enhanced Byzantine Detection
pub use pogq::{
    PoGQv41Config, PoGQv41Enhanced, PoGQv41Lite,
    GradientResult, GradientScores, Phase2Status,
    MondrianProfile, LambdaStats, AggregationDiagnostics,
    AdaptiveHybridScorer, MondrianConformal,
    DetectionResult as PoGQDetectionResult,
};

// Streaming aggregation for gradients larger than memory
pub use streaming::{
    StreamingConfig, StreamingAggregator, StreamingResult,
    ChunkedGradient, SubmissionHandle, StreamingProgress, StreamingPhase,
    StreamingCheckpoint, StreamingKrumApproximator,
    aggregate_chunks,
};

// Round Coordinator for multi-round FL
pub use coordinator::{
    RoundCoordinator, CoordinatorConfig, CoordinatorEvent, CoordinatorCommand,
    RoundState, RoundInfo,
};

// gRPC service (when feature enabled)
#[cfg(feature = "grpc")]
pub use grpc::{
    FederatedLearningServer, FederatedLearningService,
    ServiceState, GrpcServiceConfig, NodeSession,
    // Proto stub types
    GrpcPayload, GrpcPayloadType, GrpcPayloadData,
    GrpcDenseGradient, GrpcSparseGradient, GrpcQuantizedGradient,
    GrpcHyperEncodedGradient, GrpcBinaryHypervector,
    RegisterNodeRequest, RegisterNodeResponse,
    SubmitRequest, SubmitResponse,
    GetStatusRequest, GetStatusResponse,
    GetAggregatedRequest, GetAggregatedResponse,
    RoundEvent, GrpcRoundEventType, GrpcRoundState,
    NodeIdentity, NodeCapabilities, Signature,
};

/// Gradient type alias
pub type Gradient = ndarray::Array1<f32>;

/// Node identifier
pub type NodeId = String;

/// Round number
pub type Round = u64;

// Gradient Replay Detection
pub use replay::{
    ReplayDetector, ReplayDetectorConfig, ReplayDetectorStats,
    GradientFingerprint, ReplayCheckResult, ReplayType,
    SuspiciousNode, cosine_similarity,
};

// Shapley Value Attribution for fair contribution scoring
pub use shapley::{
    ShapleyCalculator, ShapleyConfig, ShapleyResult,
    SamplingMethod, Baseline, ValueFunction,
    verify_efficiency, verify_symmetry, verify_null_player,
};

// Prometheus metrics (when feature enabled)
#[cfg(feature = "prometheus")]
pub use prometheus::{FLMetrics, RoundMetrics};

#[cfg(all(feature = "prometheus", feature = "http-api"))]
pub use prometheus::MetricsMiddleware;

// WebAssembly bindings (when feature enabled)
#[cfg(feature = "wasm")]
pub use wasm::{
    WasmAggregator, WasmDefense, WasmPhiMeasurer,
    WasmHyperFeel, WasmShapley, WasmBatchAggregator,
    init_logging, version as wasm_version,
    cosine_similarity_wasm, l2_norm,
};

// End-to-end demo
pub use demo::{
    FLSimulator, SimulationConfig, SimulationReport,
    RoundStats, run_quick_demo,
};
