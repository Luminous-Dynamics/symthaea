// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea–Mycelix Bridge
//!
//! This crate defines a small, explicit interface between:
//! - Symthaea-HLB's consciousness engine (Φ, HDC, epistemic metrics)
//! - Mycelix SDK's canonical Epistemic + MATL + HyperFeel types
//!
//! It is intentionally minimal and focused on:
//! - Quality assessment of gradients / hypergradients
//! - Connectivity-based anomaly detection hooks
//! - Mapping results into `mycelix_sdk::epistemic` classifications

pub mod fl_plugin;
pub mod support;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

use mycelix_sdk::epistemic::{ClaimBuilder, EpistemicClaim};
use mycelix_sdk::hyperfeel::{HV16_BYTES, HyperGradient};
use mycelix_sdk::matl::ProofOfGradientQuality;

use symthaea_core::consciousness_metrics::TruePhiCalculator;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

// ============================================================================
// LOCAL TYPES — Inlined from symthaea::mycelix::{mapper, types} to avoid
// depending on the full symthaea crate (which would pull in burn, etc.).
// ============================================================================

/// Workspace scope for normative level mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum WorkspaceScope {
    Internal,
    Local,
    Network,
    Universal,
}

/// Local E/N/M classification (mirrors symthaea::mycelix::types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalEpistemicClassification {
    empirical: LocalEmpiricalLevel,
    normative: LocalNormativeLevel,
    materiality: LocalMaterialityLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalEmpiricalLevel {
    Subjective,
    Testimonial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalNormativeLevel {
    Internal,
    Agent,
    Network,
    Foundational,
}

impl LocalNormativeLevel {
    fn from_scope(scope: WorkspaceScope) -> Self {
        match scope {
            WorkspaceScope::Internal => Self::Internal,
            WorkspaceScope::Local => Self::Agent,
            WorkspaceScope::Network => Self::Network,
            WorkspaceScope::Universal => Self::Foundational,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalMaterialityLevel {
    Ephemeral,
    ShortTerm,
    MediumTerm,
    Permanent,
}

impl LocalMaterialityLevel {
    fn from_importance(importance: f32) -> Self {
        if importance >= 0.75 {
            Self::Permanent
        } else if importance >= 0.50 {
            Self::MediumTerm
        } else if importance >= 0.25 {
            Self::ShortTerm
        } else {
            Self::Ephemeral
        }
    }
}

/// Convert local classification to SDK classification for claim building.
impl LocalEpistemicClassification {
    fn to_sdk(
        &self,
    ) -> (
        mycelix_sdk::epistemic::EmpiricalLevel,
        mycelix_sdk::epistemic::NormativeLevel,
        mycelix_sdk::epistemic::MaterialityLevel,
    ) {
        let empirical = match self.empirical {
            LocalEmpiricalLevel::Subjective => mycelix_sdk::epistemic::EmpiricalLevel::E0Null,
            LocalEmpiricalLevel::Testimonial => {
                mycelix_sdk::epistemic::EmpiricalLevel::E1Testimonial
            }
        };
        let normative = match self.normative {
            LocalNormativeLevel::Internal => mycelix_sdk::epistemic::NormativeLevel::N0Personal,
            LocalNormativeLevel::Agent => mycelix_sdk::epistemic::NormativeLevel::N1Communal,
            LocalNormativeLevel::Network => mycelix_sdk::epistemic::NormativeLevel::N2Network,
            LocalNormativeLevel::Foundational => {
                mycelix_sdk::epistemic::NormativeLevel::N3Axiomatic
            }
        };
        let materiality = match self.materiality {
            LocalMaterialityLevel::Ephemeral => {
                mycelix_sdk::epistemic::MaterialityLevel::M0Ephemeral
            }
            LocalMaterialityLevel::ShortTerm => {
                mycelix_sdk::epistemic::MaterialityLevel::M1Temporal
            }
            LocalMaterialityLevel::MediumTerm => {
                mycelix_sdk::epistemic::MaterialityLevel::M2Persistent
            }
            LocalMaterialityLevel::Permanent => {
                mycelix_sdk::epistemic::MaterialityLevel::M3Foundational
            }
        };
        (empirical, normative, materiality)
    }
}

/// Maps observed evidence into E/N/M coordinates.
///
/// Connectivity is a model output, not evidence provenance. A validation run
/// performed by this node therefore supports an E1 testimonial claim only;
/// stronger levels require replay artifacts, cryptographic verification, or a
/// public reproduction protocol that this bridge does not currently accept.
struct EvidenceToEpistemicMapper;

impl EvidenceToEpistemicMapper {
    fn new() -> Self {
        Self
    }

    fn classify(
        &self,
        scope: WorkspaceScope,
        importance: f32,
        validation_observed: bool,
    ) -> LocalEpistemicClassification {
        let empirical = if validation_observed {
            LocalEmpiricalLevel::Testimonial
        } else {
            LocalEmpiricalLevel::Subjective
        };
        LocalEpistemicClassification {
            empirical,
            normative: LocalNormativeLevel::from_scope(scope),
            materiality: LocalMaterialityLevel::from_importance(importance),
        }
    }
}

// ============================================================================
// VECTOR STORE — Simple content-addressable memory with cosine similarity
// recall. Replaces HebbianAssociativeMemory for the bridge's use case.
// ============================================================================

/// Simple vector store for gradient prototype recall.
struct VectorStore {
    concepts: Vec<(String, Vec<f32>)>,
}

/// Stats for the vector store.
struct VectorStoreStats {
    num_concepts: usize,
}

impl VectorStore {
    fn new() -> Self {
        Self {
            concepts: Vec::new(),
        }
    }

    fn store(&mut self, id: &str, vector: Vec<f32>) {
        // Update existing or append new
        if let Some(entry) = self.concepts.iter_mut().find(|(k, _)| k == id) {
            entry.1 = vector;
        } else {
            self.concepts.push((id.to_string(), vector));
        }
    }

    /// Recall the most similar stored vector by cosine similarity.
    /// Returns (id, vector, similarity) or None if empty.
    fn recall_by_vector(&self, query: &[f32]) -> Option<(String, Vec<f32>, f32)> {
        if self.concepts.is_empty() {
            return None;
        }
        let q_norm = dot_norm(query);
        if q_norm < 1e-12 {
            return None;
        }
        let mut best: Option<(usize, f32)> = None;
        for (i, (_id, vec)) in self.concepts.iter().enumerate() {
            let v_norm = dot_norm(vec);
            if v_norm < 1e-12 {
                continue;
            }
            let dot: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            let sim = dot / (q_norm * v_norm);
            if best.map_or(true, |(_, s)| sim > s) {
                best = Some((i, sim));
            }
        }
        best.map(|(i, sim)| {
            let (id, vec) = &self.concepts[i];
            (id.clone(), vec.clone(), sim)
        })
    }

    fn stats(&self) -> VectorStoreStats {
        VectorStoreStats {
            num_concepts: self.concepts.len(),
        }
    }
}

fn dot_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Error type for bridge operations.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// Underlying Symthaea error.
    #[error("Symthaea error: {0}")]
    Symthaea(String),

    /// Underlying Mycelix error.
    #[error("Mycelix error: {0}")]
    Mycelix(String),

    /// Unsupported or unimplemented operation.
    #[error("Operation not implemented: {0}")]
    NotImplemented(&'static str),
}

/// Result type for bridge operations.
pub type Result<T> = std::result::Result<T, BridgeError>;

/// Map a conscious quality score into a MATL PoGQ signal.
///
/// This adapter lets downstream systems integrate Symthaea's assessment
/// into MATL's ProofOfGradientQuality without depending on the internal
/// anomaly heuristics.
pub fn pogq_from_quality_score(q: &QualityScore) -> ProofOfGradientQuality {
    if q.validate().is_err() {
        // Preserve the infallible adapter API while ensuring malformed or
        // non-finite scores can never turn into a permissive MATL signal.
        return ProofOfGradientQuality::new(0.0, 0.0, 1.0);
    }

    // Treat epistemic confidence as the primary quality signal.
    let quality = q.epistemic_confidence as f64;

    // Consistency captures both connectivity trend and similarity to prior behavior.
    let connectivity_trend = if q.spectral.connectivity_gain >= 0.0 {
        1.0
    } else {
        (1.0 - (-q.spectral.connectivity_gain as f64)).clamp(0.0, 1.0)
    };

    let sim_component = q.similarity.unwrap_or(0.0) as f64;
    let consistency = 0.5 * connectivity_trend + 0.5 * sim_component;

    // Entropy: ambiguous or anomalous updates have higher "uncertainty".
    let entropy = if q.is_anomalous {
        match q.severity {
            ConsciousAnomalySeverity::Severe => 1.0,
            ConsciousAnomalySeverity::Moderate => 0.7,
            ConsciousAnomalySeverity::Mild => 0.4,
            ConsciousAnomalySeverity::None => 0.2,
        }
    } else if q.is_ambiguous {
        0.5
    } else {
        0.1
    };

    ProofOfGradientQuality::new(quality, consistency, entropy)
}

/// Convert a 32-byte gradient hash into a stable hex identifier.
fn gradient_id_from_hash(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", byte);
    }
    s
}

/// Spectral connectivity assessment from Symthaea's PhiEngine.
///
/// Uses algebraic connectivity (Fiedler value) of the network graph.
/// NOT true IIT Phi — SpectralConnectivity λ₂ has r≈-0.14 vs ExhaustivePartition tier.
/// See `symthaea-core/src/consciousness_metrics/spectral_mip.rs` for the
/// production SpectralMIPFinder (MI Laplacian + MIP sweep, separate algorithm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralConnectivityAssessment {
    /// Connectivity score before applying the update.
    pub connectivity_before: f32,
    /// Connectivity score after applying the update.
    pub connectivity_after: f32,
    /// Difference (positive = more connectivity/coherence).
    pub connectivity_gain: f32,
}

/// Deprecated: use [`SpectralConnectivityAssessment`] instead.
pub type IntegrationAssessment = SpectralConnectivityAssessment;

/// Multi-dimensional consciousness assessment (C-Vector).
///
/// Each dimension is `Option<f64>` for incremental population —
/// not all dimensions are computed every cycle.
/// Use [`composite()`](ConsciousnessVector::composite) for a backward-compatible single scalar.
///
/// See `CONSCIOUSNESS_METRICS.md` for what each dimension actually measures.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsciousnessVector {
    /// Spectral connectivity (Fiedler value) [0,1]. Always computed. NOT IIT Phi.
    pub spectral_connectivity: Option<f64>,
    /// True IIT Phi from TruePhiCalculator [0,1]. Expensive: O(2^n) for n<=8.
    pub true_phi: Option<f64>,
    /// Fast Phi approximation via effective information [0,1]. O(n^2).
    pub phi_fast: Option<f64>,
    /// Shannon entropy of state space [0,1] (normalized).
    pub entropy: Option<f64>,
    /// Internal coherence: 1 - mean pairwise distance [0,1].
    pub coherence: Option<f64>,
    /// Epistemic confidence from validation metrics [0,1].
    pub epistemic_confidence: Option<f64>,
}

impl ConsciousnessVector {
    /// Best available phi: true_phi > phi_fast > spectral_connectivity.
    pub fn best_phi(&self) -> f64 {
        self.true_phi
            .or(self.phi_fast)
            .or(self.spectral_connectivity)
            .filter(|value| value.is_finite())
            .unwrap_or(0.0)
    }

    /// Whether every populated dimension is finite and lies in [0, 1].
    pub fn is_valid(&self) -> bool {
        [
            self.spectral_connectivity,
            self.true_phi,
            self.phi_fast,
            self.entropy,
            self.coherence,
            self.epistemic_confidence,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }

    /// Weighted composite for backward compatibility.
    /// Weights: phi=0.35, coherence=0.20, entropy=0.15, epistemic=0.15, spectral=0.15
    ///
    /// Only populated dimensions contribute; result is renormalized.
    pub fn composite(&self) -> f64 {
        if !self.is_valid() {
            return 0.0;
        }

        let mut total = 0.0;
        let mut weight_sum = 0.0;

        if let Some(v) = self.true_phi.or(self.phi_fast) {
            total += 0.35 * v;
            weight_sum += 0.35;
        }
        if let Some(v) = self.coherence {
            total += 0.20 * v;
            weight_sum += 0.20;
        }
        if let Some(v) = self.entropy {
            total += 0.15 * v;
            weight_sum += 0.15;
        }
        if let Some(v) = self.epistemic_confidence {
            total += 0.15 * v;
            weight_sum += 0.15;
        }
        if let Some(v) = self.spectral_connectivity {
            total += 0.15 * v;
            weight_sum += 0.15;
        }

        if weight_sum > 0.0 {
            (total / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Number of populated (Some) dimensions.
    pub fn populated_count(&self) -> usize {
        [
            self.spectral_connectivity.is_some(),
            self.true_phi.is_some(),
            self.phi_fast.is_some(),
            self.entropy.is_some(),
            self.coherence.is_some(),
            self.epistemic_confidence.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }
}

/// Configuration for C-Vector computation in the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessVectorConfig {
    /// Maximum component count for true_phi computation (default: 4).
    /// Set to 0 to disable true_phi entirely.
    pub max_components_true_phi: usize,
    /// Whether to compute phi_fast (default: true).
    pub compute_phi_fast: bool,
    /// Max entropy value for [0,1] normalization (log2 of num_bins).
    pub max_entropy: f64,
}

impl Default for ConsciousnessVectorConfig {
    fn default() -> Self {
        Self {
            max_components_true_phi: 4,
            compute_phi_fast: true,
            // log2(16) = 4.0 for default 16-bin entropy
            max_entropy: 4.0,
        }
    }
}

impl SpectralConnectivityAssessment {
    /// Construct a new assessment from before/after values.
    pub fn new(connectivity_before: f32, connectivity_after: f32) -> Self {
        Self {
            connectivity_before,
            connectivity_after,
            connectivity_gain: connectivity_after - connectivity_before,
        }
    }
}

/// Consciousness-guided quality score for a model update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Validation accuracy after applying the update.
    pub accuracy: f32,
    /// Validation loss after applying the update.
    pub loss: f32,
    /// Spectral connectivity assessment (before/after/gain).
    pub spectral: SpectralConnectivityAssessment,
    /// Multi-dimensional consciousness vector.
    pub consciousness_vector: ConsciousnessVector,
    /// Epistemic confidence in the quality claim [0, 1].
    pub epistemic_confidence: f32,
    /// Whether the update appears anomalous (potentially Byzantine).
    pub is_anomalous: bool,
    /// Maximum similarity to any stored prototype (if recall attempted).
    pub similarity: Option<f32>,
    /// Whether this update lies in the "gray zone" (ambiguous recall).
    pub is_ambiguous: bool,
    /// Conscious anomaly severity classification.
    pub severity: ConsciousAnomalySeverity,
    /// Machine-readable reasons for anomaly classification (if any).
    /// Example: ["phi_drop", "gray_zone"].
    pub causes: Vec<String>,
}

impl QualityScore {
    /// Validate all numeric inputs before they influence aggregation weights or
    /// signed attestations.
    pub fn validate(&self) -> Result<()> {
        if !self.accuracy.is_finite() || !(0.0..=1.0).contains(&self.accuracy) {
            return Err(BridgeError::Mycelix(
                "quality accuracy must be finite and in [0, 1]".into(),
            ));
        }
        if !self.loss.is_finite() || self.loss < 0.0 {
            return Err(BridgeError::Mycelix(
                "quality loss must be finite and non-negative".into(),
            ));
        }
        if !self.epistemic_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.epistemic_confidence)
        {
            return Err(BridgeError::Mycelix(
                "epistemic confidence must be finite and in [0, 1]".into(),
            ));
        }
        if !self.spectral.connectivity_before.is_finite()
            || !self.spectral.connectivity_after.is_finite()
            || !self.spectral.connectivity_gain.is_finite()
            || !(0.0..=1.0).contains(&self.spectral.connectivity_before)
            || !(0.0..=1.0).contains(&self.spectral.connectivity_after)
        {
            return Err(BridgeError::Mycelix(
                "spectral connectivity must be finite and in [0, 1]".into(),
            ));
        }
        let expected_gain = self.spectral.connectivity_after - self.spectral.connectivity_before;
        if (self.spectral.connectivity_gain - expected_gain).abs() > 1e-5 {
            return Err(BridgeError::Mycelix(
                "spectral connectivity gain is inconsistent".into(),
            ));
        }
        if self
            .similarity
            .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
        {
            return Err(BridgeError::Mycelix(
                "similarity must be finite and in [-1, 1]".into(),
            ));
        }
        if !self.consciousness_vector.is_valid() {
            return Err(BridgeError::Mycelix(
                "consciousness vector values must be finite and in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Severity classification for conscious anomaly detection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsciousAnomalySeverity {
    /// No anomaly detected.
    None,
    /// Mild concern (e.g., gray-zone similarity without strong Φ drop).
    Mild,
    /// Clear anomaly (Φ drop or very low confidence).
    Moderate,
    /// Strong anomaly (large Φ drop and/or extremely low confidence).
    Severe,
}

/// Minimal abstraction over "model + test data" so that different
/// backends (PyTorch, Symthaea LTC, ONNX, etc.) can be plugged in.
pub trait ModelSnapshot {
    /// Apply an encoded update (e.g. HyperFeel hypergradient) to a clone
    /// of the model, leaving the original untouched.
    fn apply_update_clone(
        &self,
        update: &mycelix_sdk::hyperfeel::HyperGradient,
    ) -> Result<Box<dyn ModelSnapshot>>;

    /// Evaluate the snapshot on a validation set, returning (accuracy, loss).
    fn evaluate(&self) -> Result<(f32, f32)>;

    /// Current validation accuracy (if known) before applying updates.
    fn current_accuracy(&self) -> Option<f32>;
}

/// Trait for a backend that can provide Φ and quality assessments for
/// Mycelix FL updates.
pub trait ConsciousnessBackend {
    /// Assess the quality of a HyperFeel-encoded gradient / update.
    ///
    /// Implementations are expected to:
    /// - compute Φ before and after applying the update,
    /// - evaluate validation accuracy/loss,
    /// - derive an epistemic confidence and anomaly flag,
    /// - and optionally map into `mycelix_sdk::epistemic::EpistemicClaim`
    ///   in higher-level code.
    fn assess_update(
        &mut self,
        snapshot: &dyn ModelSnapshot,
        update: &HyperGradient,
    ) -> Result<QualityScore>;
}

/// Symthaea-backed implementation of the consciousness backend.
///
/// This implementation:
/// - interprets the incoming HyperFeel `HyperGradient` as a 16,384D
///   continuous hypervector via a sparse random projection,
/// - uses `PhiEngine` to compute a simple Φ score over a 1-node
///   "topology" (the update in isolation),
/// - records the local validation as testimonial evidence without inferring
///   stronger provenance from Φ,
/// - combines it with validation metrics from the `ModelSnapshot` to
///   produce a `QualityScore`.
pub struct SymthaeaBackend {
    config: SymthaeaBackendConfig,
    phi_engine: PhiEngine,
    mapper: EvidenceToEpistemicMapper,
    /// Vector store for gradient prototypes (content-addressable cosine recall).
    memory: VectorStore,
    /// Per-node state for trend-aware connectivity tracking and anomaly counts.
    node_states: HashMap<String, NodeState>,
    /// Shared sparse projector for HyperGradient → ContinuousHV.
    projector: SparseProjector,
    /// True IIT Phi calculator for C-Vector computation.
    true_phi_calculator: TruePhiCalculator,
    /// Configuration for C-Vector computation.
    cvector_config: ConsciousnessVectorConfig,
}

/// Configuration for the Symthaea backend behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymthaeaBackendConfig {
    /// Similarity above which we snap to a stored prototype.
    pub recall_threshold: f32,
    /// Lower bound of the "gray zone" where updates are considered ambiguous.
    pub ambiguity_threshold: f32,
    /// Minimum connectivity drop (before → after) to consider suspicious.
    pub connectivity_drop_threshold: f32,
    /// Optional cap on number of concepts stored in associative memory.
    /// `None` = unbounded (not recommended for long-running deployments).
    pub max_concepts: Option<usize>,
}

impl Default for SymthaeaBackendConfig {
    fn default() -> Self {
        Self {
            recall_threshold: 0.9,
            ambiguity_threshold: 0.75,
            connectivity_drop_threshold: 0.1,
            max_concepts: Some(10_000),
        }
    }
}

impl SymthaeaBackendConfig {
    /// Strict mode: sensitive to Φ drops and gray-zone behavior.
    pub fn strict() -> Self {
        Self {
            recall_threshold: 0.92,
            ambiguity_threshold: 0.8,
            connectivity_drop_threshold: 0.05,
            max_concepts: Some(5_000),
        }
    }

    /// Lenient mode: tolerant to fluctuations, good for exploratory training.
    pub fn lenient() -> Self {
        Self {
            recall_threshold: 0.88,
            ambiguity_threshold: 0.7,
            connectivity_drop_threshold: 0.15,
            max_concepts: Some(20_000),
        }
    }

    /// Diagnostic mode: surfaces many gray-zone events for analysis.
    pub fn diagnostic() -> Self {
        Self {
            recall_threshold: 0.9,
            ambiguity_threshold: 0.6,
            connectivity_drop_threshold: 0.05,
            max_concepts: Some(2_000),
        }
    }
}

/// Maximum number of recent HVs stored per node for multi-component C-Vector.
const NODE_HV_HISTORY_CAP: usize = 4;

/// Per-node consciousness state.
#[derive(Debug, Clone)]
struct NodeState {
    /// Last observed connectivity score for this node.
    last_connectivity: f32,
    /// Number of anomalous updates detected so far.
    anomaly_count: u32,
    /// Recent HVs for this node (bounded ring buffer).
    /// Used as the multi-component system for true_phi and coherence.
    recent_hvs: VecDeque<ContinuousHV>,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            last_connectivity: 0.5, // Neutral starting point
            anomaly_count: 0,
            recent_hvs: VecDeque::with_capacity(NODE_HV_HISTORY_CAP),
        }
    }
}

/// Sparse random projector from low-dimensional byte space (HV16 bytes)
/// to Symthaea's continuous HDC space (`HDC_DIMENSION`).
///
/// This implements a lightweight Johnson–Lindenstrauss style projection
/// with a small, fixed fan-out per input dimension so we do not need to
/// store the full projection matrix.
pub struct SparseProjector {
    /// Number of non-zero targets per input dimension.
    fan_out: usize,
    /// Output dimension (should match `HDC_DIMENSION`).
    output_dim: usize,
}

impl SparseProjector {
    /// Create a new projector with the given fan-out and output dimension.
    pub fn new(fan_out: usize, output_dim: usize) -> Self {
        Self {
            fan_out,
            output_dim,
        }
    }

    /// Deterministically project a byte vector into a normalized ContinuousHV.
    pub fn project(&self, bytes: &[u8]) -> ContinuousHV {
        let mut values = vec![0.0f32; self.output_dim];

        for (i, &b) in bytes.iter().enumerate() {
            // Map byte [0,255] → [-1, 1]
            let base = (b as f32 / 255.0) * 2.0 - 1.0;

            // Seed PRNG from index so projection is deterministic per coordinate.
            let mut state = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

            for _ in 0..self.fan_out {
                // Simple xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                let idx = (state as usize) % self.output_dim;
                let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };

                values[idx] += base * sign;
            }
        }

        ContinuousHV::from_vec(values).normalize()
    }
}

impl SymthaeaBackend {
    /// Create a new backend with default configuration (continuous Φ).
    pub fn new() -> Self {
        Self::with_config(SymthaeaBackendConfig::default())
    }

    /// Create a new backend with an explicit configuration.
    pub fn with_config(config: SymthaeaBackendConfig) -> Self {
        Self::with_full_config(config, ConsciousnessVectorConfig::default())
    }

    /// Create with both backend and C-Vector configuration.
    pub fn with_full_config(
        config: SymthaeaBackendConfig,
        cvector_config: ConsciousnessVectorConfig,
    ) -> Self {
        let phi_engine = PhiEngine::new(PhiMethod::SpectralConnectivity);
        let mapper = EvidenceToEpistemicMapper::new();
        Self {
            config,
            phi_engine,
            mapper,
            memory: VectorStore::new(),
            node_states: HashMap::new(),
            projector: SparseProjector::new(8, HDC_DIMENSION),
            true_phi_calculator: TruePhiCalculator::new(),
            cvector_config,
        }
    }

    /// Reset state for a single node (clears Φ history and anomaly count).
    pub fn reset_node(&mut self, node_id: &str) {
        self.node_states.remove(node_id);
    }

    /// Reset all backend state, including node histories and associative memory.
    pub fn reset_all(&mut self) {
        self.node_states.clear();
        self.memory = VectorStore::new();
    }

    /// Get a lightweight snapshot of a node's state (last connectivity, anomaly count).
    pub fn node_state_snapshot(&self, node_id: &str) -> Option<(f32, u32)> {
        self.node_states
            .get(node_id)
            .map(|s| (s.last_connectivity, s.anomaly_count))
    }

    /// Access the current configuration.
    pub fn config(&self) -> &SymthaeaBackendConfig {
        &self.config
    }

    /// Mutably access the configuration for runtime tuning.
    pub fn config_mut(&mut self) -> &mut SymthaeaBackendConfig {
        &mut self.config
    }

    /// Helper: convert a HyperFeel hypervector (bytes) into a normalized
    /// Symthaea `ContinuousHV` using a sparse random projection:
    /// - bytes in [0, 255] are mapped linearly to [-1, 1]
    /// - projected into HDC space using a fixed-fan-out sparse projector
    fn hypergradient_to_hv(&self, update: &HyperGradient) -> Result<ContinuousHV> {
        if update.hypervector.len() != HV16_BYTES {
            return Err(BridgeError::Mycelix(format!(
                "HyperGradient hypervector length {}, expected {}",
                update.hypervector.len(),
                HV16_BYTES
            )));
        }

        Ok(self.projector.project(&update.hypervector))
    }

    /// Recall-or-project: try to snap the projected HV to a known prototype
    /// using Hebbian associative memory; otherwise treat as novel and store.
    ///
    /// Returns both the hypervector used for Φ calculation and the maximum
    /// similarity to any stored prototype (if recall was attempted).
    fn recall_or_project(&mut self, update: &HyperGradient) -> Result<(ContinuousHV, Option<f32>)> {
        let projected = self.hypergradient_to_hv(update)?;

        let mut best_sim: Option<f32> = None;
        let mut best_vec: Option<Vec<f32>> = None;

        // 1. Try vector-based recall from associative memory.
        if let Some((_id, proto_vec, similarity)) = self.memory.recall_by_vector(&projected.values)
        {
            best_sim = Some(similarity);
            best_vec = Some(proto_vec);
        }

        let hv = if let Some(sim) = best_sim {
            if sim >= self.config.recall_threshold {
                ContinuousHV::from_vec(best_vec.expect("prototype vector present"))
            } else {
                projected.clone()
            }
        } else {
            projected.clone()
        };

        // 2. Store (or reinforce) a concept under a stable ID derived from hash,
        //    respecting an optional capacity limit.
        if self
            .config
            .max_concepts
            .map(|limit| self.memory.stats().num_concepts < limit)
            .unwrap_or(true)
        {
            let id = gradient_id_from_hash(&update.gradient_hash);
            self.memory.store(&id, projected.values.clone());
        }

        Ok((hv, best_sim))
    }

    /// Derive an epistemic classification from the observed validation and FL context.
    fn classify_from_validation(
        &self,
        _connectivity: f32,
        update: &HyperGradient,
    ) -> LocalEpistemicClassification {
        // WorkspaceScope is not directly known here; treat FL updates as
        // network-level by default.
        let scope = WorkspaceScope::Network;

        // Importance: use a simple heuristic based on gradient magnitude
        // (quality_score is L2 norm) and compression ratio.
        let raw_importance = update.quality_score / (update.compression_ratio + 1.0);
        let importance = if raw_importance.is_finite() {
            raw_importance.clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.mapper.classify(scope, importance, true)
    }

    /// Helper: map Symthaea's classification into a canonical Mycelix claim.
    fn build_claim(
        &self,
        phi: f32,
        snapshot: &dyn ModelSnapshot,
        classification: &LocalEpistemicClassification,
    ) -> EpistemicClaim {
        let acc_before = snapshot
            .current_accuracy()
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .unwrap_or(0.0);
        let content = format!(
            "Symthaea assessed FL update: Φ={:.3}, accuracy_before≈{:.3}",
            phi, acc_before
        );

        let (empirical, normative, materiality) = classification.to_sdk();

        ClaimBuilder::new(content)
            .empirical(empirical)
            .normative(normative)
            .materiality(materiality)
            .build()
    }

    /// Compute the multi-dimensional ConsciousnessVector.
    ///
    /// `components` is the node's recent HV history (most recent last).
    /// With ≥2 components, true_phi and coherence become non-trivial.
    fn compute_consciousness_vector(
        &self,
        components: &[ContinuousHV],
        spectral_connectivity: f64,
        epistemic_confidence: f64,
    ) -> ConsciousnessVector {
        // Use the most recent HV for entropy (single-component metric)
        let current_hv = components.last().expect("components must be non-empty");

        // Entropy: normalized Shannon entropy via TruePhiCalculator
        let raw_entropy = self.true_phi_calculator.entropy(current_hv);
        let max_e = self.cvector_config.max_entropy;
        let entropy = if max_e > 0.0 {
            (raw_entropy / max_e).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Phi fast: effective information approximation (O(n^2) in component count)
        let phi_fast = if self.cvector_config.compute_phi_fast {
            Some(self.true_phi_calculator.compute_phi_fast(components))
        } else {
            None
        };

        // True phi: only for small component counts (default ≤ 4)
        let true_phi = if components.len() >= 2
            && components.len() <= self.cvector_config.max_components_true_phi
        {
            let result = self.true_phi_calculator.compute_true_phi(components);
            Some(result.phi.clamp(0.0, 1.0))
        } else {
            None
        };

        // Coherence: for a single HV, self-coherence is 1.0 (trivial).
        // For multi-component systems, use mean pairwise cosine similarity.
        let coherence = if components.len() <= 1 {
            1.0
        } else {
            let mut total_sim = 0.0;
            let mut pairs = 0;
            for i in 0..components.len() {
                for j in (i + 1)..components.len() {
                    total_sim += components[i].similarity(&components[j]) as f64;
                    pairs += 1;
                }
            }
            if pairs > 0 {
                (total_sim / pairs as f64).clamp(0.0, 1.0)
            } else {
                1.0
            }
        };

        ConsciousnessVector {
            spectral_connectivity: Some(spectral_connectivity),
            true_phi,
            phi_fast,
            entropy: Some(entropy),
            coherence: Some(coherence),
            epistemic_confidence: Some(epistemic_confidence),
        }
    }
}

impl Default for SymthaeaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessBackend for SymthaeaBackend {
    fn assess_update(
        &mut self,
        snapshot: &dyn ModelSnapshot,
        update: &HyperGradient,
    ) -> Result<QualityScore> {
        // Validate the transport shape before applying untrusted update data to
        // a model backend or mutating bridge state.
        if update.node_id.trim().is_empty() {
            return Err(BridgeError::Mycelix(
                "HyperGradient node_id must not be empty".into(),
            ));
        }
        self.hypergradient_to_hv(update)?;

        // Evaluate a clone with the update applied. The original snapshot is
        // intentionally retained for the before-update claim metadata.
        let updated_snapshot = snapshot.apply_update_clone(update)?;
        let (accuracy, loss) = updated_snapshot.evaluate()?;
        if !accuracy.is_finite() || !(0.0..=1.0).contains(&accuracy) {
            return Err(BridgeError::Mycelix(
                "updated snapshot accuracy must be finite and in [0, 1]".into(),
            ));
        }
        if !loss.is_finite() || loss < 0.0 {
            return Err(BridgeError::Mycelix(
                "updated snapshot loss must be finite and non-negative".into(),
            ));
        }

        // 1. Identify node and load prior state (extract connectivity_before, drop borrow)
        let node_id = update.node_id.to_string();
        let connectivity_before = self
            .node_states
            .entry(node_id.clone())
            .or_insert_with(NodeState::default)
            .last_connectivity;

        // 2. Convert HyperGradient -> ContinuousHV, with recall-or-project semantics.
        let (hv, similarity) = self.recall_or_project(update)?;

        // 3. Compute spectral connectivity score for this node
        let phi_result = self.phi_engine.compute(&[hv.clone()]);
        let connectivity_after = phi_result.phi as f32;
        if !connectivity_after.is_finite() || !(0.0..=1.0).contains(&connectivity_after) {
            return Err(BridgeError::Symthaea(
                "spectral connectivity must be finite and in [0, 1]".into(),
            ));
        }
        let spectral_assessment =
            SpectralConnectivityAssessment::new(connectivity_before, connectivity_after);

        // 4. Record the locally-observed validation at an honest evidence level.
        let classification = self.classify_from_validation(connectivity_after, update);

        // 6. Build a canonical EpistemicClaim (currently unused, but ready
        //    for callers that want to store it in Mycelix UESS)
        let claim = self.build_claim(connectivity_after, snapshot, &classification);
        let _claim_code = claim.code(); // exercise the claim for now

        // 7. Epistemic confidence: scaled combination of connectivity and accuracy
        let epistemic_confidence = ((connectivity_after + accuracy) / 2.0).clamp(0.0, 1.0);

        // 7b. Compute the prospective C-Vector without committing node state
        // until the complete quality result has passed validation.
        let mut components: Vec<ContinuousHV> = self.node_states[&node_id]
            .recent_hvs
            .iter()
            .cloned()
            .collect();
        if components.len() >= NODE_HV_HISTORY_CAP {
            components.remove(0);
        }
        components.push(hv);
        let consciousness_vector = self.compute_consciousness_vector(
            &components,
            connectivity_after as f64,
            epistemic_confidence as f64,
        );

        // 8. Anomaly heuristic:
        //    - strong negative connectivity gain
        //    - or gray-zone similarity (uncanny valley)
        //    - or very low epistemic confidence
        let connectivity_gain = spectral_assessment.connectivity_gain;

        let is_ambiguous = similarity.map_or(false, |sim| {
            sim >= self.config.ambiguity_threshold && sim < self.config.recall_threshold
        });

        let is_drop = connectivity_gain < -self.config.connectivity_drop_threshold;
        let low_confidence = epistemic_confidence < 0.2;

        let is_anomalous = is_drop || is_ambiguous || low_confidence;

        // 10. Classify severity
        let severity = if !is_anomalous {
            ConsciousAnomalySeverity::None
        } else if is_drop
            && (connectivity_gain < -2.0 * self.config.connectivity_drop_threshold
                || low_confidence)
        {
            ConsciousAnomalySeverity::Severe
        } else if is_drop || is_ambiguous || low_confidence {
            // Any single signal without the stronger criteria above.
            ConsciousAnomalySeverity::Moderate
        } else {
            ConsciousAnomalySeverity::Mild
        };

        // 11. Capture explicit causes for explainability.
        let mut causes: Vec<String> = Vec::new();
        if is_drop {
            causes.push("phi_drop".into());
        }
        if is_ambiguous {
            causes.push("gray_zone".into());
        }
        if low_confidence {
            causes.push("low_confidence".into());
        }

        let quality = QualityScore {
            accuracy,
            loss,
            spectral: spectral_assessment,
            consciousness_vector,
            epistemic_confidence,
            is_anomalous,
            similarity,
            is_ambiguous,
            severity,
            causes,
        };
        quality.validate()?;

        // Commit the prospective history and trend only after validation.
        if let Some(state) = self.node_states.get_mut(&node_id) {
            state.recent_hvs = components.into_iter().collect();
            state.last_connectivity = connectivity_after;
            if is_anomalous {
                state.anomaly_count = state.anomaly_count.saturating_add(1);
            }
        }

        Ok(quality)
    }
}

// ============================================================================
// SUPPORT BRIDGE METHODS
// ============================================================================

impl SymthaeaBackend {
    /// Triage an incoming support request and produce ticket field suggestions.
    pub fn assess_support_triage(
        &self,
        title: &str,
        description: &str,
    ) -> support::TicketFieldSuggestions {
        let engine = symthaea_support::triage::TriageEngine::new();
        let result = engine.triage(title, description);
        support::triage_to_ticket_fields(
            &format!("{:?}", result.suggested_priority),
            &format!("{:?}", result.suggested_category),
            result.confidence,
            result.suggested_articles.clone(),
            &format!("{:?}", result.epistemic_status),
        )
    }

    /// Run a diagnostic step and package the result for DHT storage.
    pub fn assess_diagnostic(
        &self,
        diagnostic_type: &str,
        findings: &str,
        severity: &str,
        recommendations: Vec<String>,
        scrubbed: bool,
    ) -> support::DiagnosticEntryData {
        support::step_to_diagnostic_entry(
            diagnostic_type,
            findings,
            severity,
            recommendations,
            scrubbed,
        )
    }

    /// Package a BinaryHV encoding as a cognitive update for DHT sharing.
    pub fn package_cognitive_update(
        &self,
        encoding: Vec<u8>,
        phi: f64,
        category: &str,
        pattern: &str,
    ) -> support::CognitiveUpdatePackage {
        support::memory_to_cognitive_update(encoding, phi, category, pattern)
    }
}

// ============================================================================
// CONSCIOUSNESS ATTESTATION GENERATION
// ============================================================================

/// Data for an authenticated consciousness attestation, ready to be signed and
/// submitted to the governance bridge via `record_consciousness_attestation()`.
///
/// NOTE: The `consciousness_level` is derived from SpectralConnectivity (Fiedler value),
/// NOT true IIT Phi. See `SpectralConnectivityAssessment` for details.
///
/// The `signature` field must be filled by the caller using their agent's
/// signing key (e.g., Holochain agent key or Ed25519 key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAttestationData {
    /// Agent's DID identifier.
    pub agent_did: String,
    /// Consciousness level from C-Vector composite, in [0.0, 1.0].
    pub consciousness_level: f64,
    /// Symthaea cognitive cycle number that produced this assessment.
    pub cycle_id: u64,
    /// Unix timestamp (microseconds) when the assessment was captured.
    pub captured_at_us: u64,
    /// Signature over the canonical sign message (filled by caller).
    pub signature: Vec<u8>,
    /// Source system identifier (always "symthaea").
    pub source: String,
    /// C-Vector dimensions for governance per-dimension gating.
    /// Serializes to the same JSON shape as governance's `ConsciousnessVectorEntry`.
    pub consciousness_vector: Option<ConsciousnessVectorSerde>,
}

/// Serializable C-Vector for governance submission.
///
/// Field names match `governance_bridge_integrity::ConsciousnessVectorEntry` exactly,
/// so `serde_json` round-trip works as a type adapter between the two crates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessVectorSerde {
    pub spectral_connectivity: Option<f64>,
    pub true_phi: Option<f64>,
    pub phi_fast: Option<f64>,
    pub entropy: Option<f64>,
    pub coherence: Option<f64>,
    pub epistemic_confidence: Option<f64>,
}

impl From<&ConsciousnessVector> for ConsciousnessVectorSerde {
    fn from(cv: &ConsciousnessVector) -> Self {
        Self {
            spectral_connectivity: cv.spectral_connectivity,
            true_phi: cv.true_phi,
            phi_fast: cv.phi_fast,
            entropy: cv.entropy,
            coherence: cv.coherence,
            epistemic_confidence: cv.epistemic_confidence,
        }
    }
}

impl ConsciousnessVectorSerde {
    fn is_valid(&self) -> bool {
        [
            self.spectral_connectivity,
            self.true_phi,
            self.phi_fast,
            self.entropy,
            self.coherence,
            self.epistemic_confidence,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

fn append_length_prefixed(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn append_optional_f64(output: &mut Vec<u8>, value: Option<f64>) {
    match value {
        Some(value) => {
            output.push(1);
            output.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        None => output.push(0),
    }
}

impl ConsciousnessAttestationData {
    /// Validate the complete signed payload.
    pub fn validate_for_signing(&self) -> Result<()> {
        if self.agent_did.trim().is_empty() {
            return Err(BridgeError::Mycelix(
                "attestation agent DID must not be empty".into(),
            ));
        }
        if self.source.trim().is_empty() {
            return Err(BridgeError::Mycelix(
                "attestation source must not be empty".into(),
            ));
        }
        if !self.consciousness_level.is_finite() || !(0.0..=1.0).contains(&self.consciousness_level)
        {
            return Err(BridgeError::Mycelix(
                "attestation consciousness level must be finite and in [0, 1]".into(),
            ));
        }
        if self.captured_at_us == 0 {
            return Err(BridgeError::Mycelix(
                "attestation timestamp must be non-zero".into(),
            ));
        }
        if self
            .consciousness_vector
            .as_ref()
            .is_some_and(|vector| !vector.is_valid())
        {
            return Err(BridgeError::Mycelix(
                "attestation vector values must be finite and in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical message bytes to sign for this attestation.
    ///
    /// Version 2 is an unambiguous binary encoding. It covers the DID,
    /// composite level, cycle, timestamp, source, and every C-Vector field.
    /// The mutable `signature` field itself is intentionally excluded.
    ///
    /// The caller should sign these bytes with their agent key and store the
    /// result in the `signature` field before submitting to governance.
    pub fn sign_message(&self) -> Result<Vec<u8>> {
        self.validate_for_signing()?;

        let mut message = b"symthaea-consciousness-attestation:v2\0".to_vec();
        append_length_prefixed(&mut message, &self.agent_did);
        message.extend_from_slice(&self.consciousness_level.to_bits().to_be_bytes());
        message.extend_from_slice(&self.cycle_id.to_be_bytes());
        message.extend_from_slice(&self.captured_at_us.to_be_bytes());
        append_length_prefixed(&mut message, &self.source);

        match &self.consciousness_vector {
            Some(vector) => {
                message.push(1);
                append_optional_f64(&mut message, vector.spectral_connectivity);
                append_optional_f64(&mut message, vector.true_phi);
                append_optional_f64(&mut message, vector.phi_fast);
                append_optional_f64(&mut message, vector.entropy);
                append_optional_f64(&mut message, vector.coherence);
                append_optional_f64(&mut message, vector.epistemic_confidence);
            }
            None => message.push(0),
        }

        Ok(message)
    }
}

/// Create an unsigned `ConsciousnessAttestationData` from a `QualityScore`.
///
/// The `consciousness_level` is the validated C-Vector composite.
/// The caller must:
/// 1. Call `.sign_message()` to get the canonical bytes
/// 2. Sign those bytes with their agent key
/// 3. Set `attestation.signature = signature_bytes`
/// 4. Submit to governance via `record_consciousness_attestation()`
///
/// # Example
///
/// ```ignore
/// let quality = backend.assess_update(&snapshot, &gradient)?;
/// let mut attestation = create_consciousness_attestation_data(&quality, "did:key:z6Mk...", 42)?;
/// let message = attestation.sign_message()?;
/// attestation.signature = my_sign_fn(&message);
/// // Submit to Holochain governance bridge...
/// ```
pub fn create_consciousness_attestation_data(
    quality: &QualityScore,
    agent_did: &str,
    cycle_id: u64,
) -> Result<ConsciousnessAttestationData> {
    quality.validate()?;
    if agent_did.trim().is_empty() {
        return Err(BridgeError::Mycelix(
            "attestation agent DID must not be empty".into(),
        ));
    }

    let now_us: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| BridgeError::Mycelix(format!("system clock error: {error}")))?
        .as_micros()
        .try_into()
        .map_err(|_| BridgeError::Mycelix("attestation timestamp exceeds u64".into()))?;

    let attestation = ConsciousnessAttestationData {
        agent_did: agent_did.to_string(),
        consciousness_level: quality.consciousness_vector.composite(),
        cycle_id,
        captured_at_us: now_us,
        signature: Vec::new(), // Caller must fill this
        source: "symthaea".to_string(),
        consciousness_vector: Some(ConsciousnessVectorSerde::from(
            &quality.consciousness_vector,
        )),
    };
    attestation.validate_for_signing()?;
    Ok(attestation)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummySnapshot;

    impl ModelSnapshot for DummySnapshot {
        fn apply_update_clone(&self, _update: &HyperGradient) -> Result<Box<dyn ModelSnapshot>> {
            Ok(Box::new(DummySnapshot))
        }

        fn evaluate(&self) -> Result<(f32, f32)> {
            Ok((0.5, 1.0))
        }

        fn current_accuracy(&self) -> Option<f32> {
            Some(0.4)
        }
    }

    struct NoopBackend;

    impl ConsciousnessBackend for NoopBackend {
        fn assess_update(
            &mut self,
            snapshot: &dyn ModelSnapshot,
            _update: &HyperGradient,
        ) -> Result<QualityScore> {
            let (accuracy, loss) = snapshot.evaluate()?;
            let spectral = SpectralConnectivityAssessment::new(0.0, 0.0);
            Ok(QualityScore {
                accuracy,
                loss,
                spectral,
                consciousness_vector: ConsciousnessVector::default(),
                epistemic_confidence: 0.0,
                is_anomalous: false,
                similarity: None,
                is_ambiguous: false,
                severity: ConsciousAnomalySeverity::None,
                causes: Vec::new(),
            })
        }
    }

    #[test]
    fn test_noop_backend_compiles_and_runs() {
        let mut backend = NoopBackend;
        let snapshot = DummySnapshot;

        let hg = HyperGradient::new(
            "node-1".to_string(),
            1,
            vec![0u8; HV16_BYTES],
            0.1,
            4_000_000,
            1.0,
            [0u8; 32],
        );

        let result = backend.assess_update(&snapshot, &hg).unwrap();
        assert_eq!(result.accuracy, 0.5);
        assert_eq!(result.loss, 1.0);
    }

    #[test]
    fn test_symthaea_backend_basic_flow() {
        let mut backend = SymthaeaBackend::new();
        let snapshot = DummySnapshot;

        let hg = HyperGradient::new(
            "node-1".to_string(),
            1,
            vec![128u8; HV16_BYTES],
            0.5,
            4_000_000,
            10.0,
            [0u8; 32],
        );

        let result = backend.assess_update(&snapshot, &hg).unwrap();
        assert!(
            result.spectral.connectivity_after >= 0.0 && result.spectral.connectivity_after <= 1.0
        );
        assert!(result.epistemic_confidence >= 0.0 && result.epistemic_confidence <= 1.0);
    }

    #[test]
    fn assessment_evaluates_the_updated_clone() {
        struct OriginalSnapshot;
        struct UpdatedSnapshot;

        impl ModelSnapshot for OriginalSnapshot {
            fn apply_update_clone(
                &self,
                _update: &HyperGradient,
            ) -> Result<Box<dyn ModelSnapshot>> {
                Ok(Box::new(UpdatedSnapshot))
            }

            fn evaluate(&self) -> Result<(f32, f32)> {
                Ok((0.2, 2.0))
            }

            fn current_accuracy(&self) -> Option<f32> {
                Some(0.2)
            }
        }

        impl ModelSnapshot for UpdatedSnapshot {
            fn apply_update_clone(
                &self,
                _update: &HyperGradient,
            ) -> Result<Box<dyn ModelSnapshot>> {
                Ok(Box::new(UpdatedSnapshot))
            }

            fn evaluate(&self) -> Result<(f32, f32)> {
                Ok((0.9, 0.1))
            }

            fn current_accuracy(&self) -> Option<f32> {
                Some(0.9)
            }
        }

        let update = HyperGradient::new(
            "node-updated".to_string(),
            1,
            vec![128u8; HV16_BYTES],
            0.5,
            4_000_000,
            10.0,
            [0u8; 32],
        );
        let result = SymthaeaBackend::new()
            .assess_update(&OriginalSnapshot, &update)
            .unwrap();

        assert_eq!(result.accuracy, 0.9);
        assert_eq!(result.loss, 0.1);
    }

    #[test]
    fn test_backend_config_presets_have_expected_strictness() {
        let strict = SymthaeaBackendConfig::strict();
        let lenient = SymthaeaBackendConfig::lenient();

        // Strict mode should use a smaller Φ drop threshold and stricter
        // ambiguity handling than lenient mode.
        assert!(
            strict.connectivity_drop_threshold < lenient.connectivity_drop_threshold,
            "strict mode should be more sensitive to connectivity drops"
        );
        assert!(
            strict.ambiguity_threshold > lenient.ambiguity_threshold,
            "strict mode should treat more updates as ambiguous"
        );

        // Diagnostic mode prioritises ambiguity surfacing over memory size.
        let diagnostic = SymthaeaBackendConfig::diagnostic();
        assert!(
            diagnostic.ambiguity_threshold <= strict.ambiguity_threshold,
            "diagnostic mode should surface more gray-zone events"
        );

        // All presets must define some maximum concept capacity.
        assert!(strict.max_concepts.is_some());
        assert!(lenient.max_concepts.is_some());
        assert!(diagnostic.max_concepts.is_some());
    }

    // ============================================================================
    // SparseProjector tests
    // ============================================================================

    #[test]
    fn sparse_projector_output_has_correct_dimension() {
        let projector = SparseProjector::new(8, HDC_DIMENSION);
        let input = vec![128u8; HV16_BYTES];
        let hv = projector.project(&input);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn sparse_projector_is_deterministic() {
        let projector = SparseProjector::new(8, HDC_DIMENSION);
        let input = vec![42u8; HV16_BYTES];
        let hv1 = projector.project(&input);
        let hv2 = projector.project(&input);
        assert!(
            hv1.similarity(&hv2) > 0.999,
            "Same input must yield identical output"
        );
    }

    #[test]
    fn sparse_projector_distinct_inputs_differ() {
        let projector = SparseProjector::new(8, HDC_DIMENSION);
        let input_a = vec![0u8; HV16_BYTES];
        let input_b = vec![255u8; HV16_BYTES];
        let hv_a = projector.project(&input_a);
        let hv_b = projector.project(&input_b);
        let sim = hv_a.similarity(&hv_b);
        assert!(
            sim < 0.99,
            "Distinct inputs should produce different projections, got sim={}",
            sim
        );
    }

    #[test]
    fn sparse_projector_output_is_normalized() {
        let projector = SparseProjector::new(8, HDC_DIMENSION);
        let input = vec![100u8; HV16_BYTES];
        let hv = projector.project(&input);
        let self_sim = hv.similarity(&hv);
        assert!(
            (self_sim - 1.0).abs() < 0.01,
            "Projected HV should be normalized, self-similarity={}",
            self_sim
        );
    }

    #[test]
    fn sparse_projector_empty_input_produces_valid_hv() {
        let projector = SparseProjector::new(8, HDC_DIMENSION);
        let input: Vec<u8> = vec![];
        let hv = projector.project(&input);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    // ============================================================================
    // VectorStore tests
    // ============================================================================

    #[test]
    fn vector_store_and_recall() {
        let mut mem = VectorStore::new();
        let concept = vec![0.1f32; HDC_DIMENSION];
        mem.store("test_concept", concept.clone());
        let (id, _proto_vec, sim) = mem
            .recall_by_vector(&concept)
            .expect("should recall stored concept");
        assert_eq!(id, "test_concept");
        assert!(
            sim > 0.99,
            "Exact input should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn vector_store_recall_missing_returns_none() {
        let mem = VectorStore::new();
        let query = vec![0.5f32; HDC_DIMENSION];
        assert!(
            mem.recall_by_vector(&query).is_none(),
            "Empty memory should return None"
        );
    }

    #[test]
    fn vector_store_distinct_concepts_recalled_correctly() {
        let mut mem = VectorStore::new();
        let concept_a = vec![1.0f32; HDC_DIMENSION];
        let concept_b = vec![-1.0f32; HDC_DIMENSION];
        mem.store("a", concept_a.clone());
        mem.store("b", concept_b.clone());

        let (id_a, _, _) = mem.recall_by_vector(&concept_a).expect("recall a");
        let (id_b, _, _) = mem.recall_by_vector(&concept_b).expect("recall b");
        assert_eq!(id_a, "a");
        assert_eq!(id_b, "b");
    }

    #[test]
    fn vector_store_stats_count_concepts() {
        let mut mem = VectorStore::new();
        assert_eq!(mem.stats().num_concepts, 0);
        mem.store("c1", vec![0.1f32; HDC_DIMENSION]);
        assert_eq!(mem.stats().num_concepts, 1);
        mem.store("c2", vec![0.2f32; HDC_DIMENSION]);
        assert_eq!(mem.stats().num_concepts, 2);
    }

    // ============================================================================
    // Validation evidence classification tests
    // ============================================================================

    #[test]
    fn validation_classification_is_testimonial() {
        let backend = SymthaeaBackend::new();
        let hg = HyperGradient::new(
            "node-1".to_string(),
            1,
            vec![128u8; HV16_BYTES],
            0.5,
            4_000_000,
            10.0,
            [0u8; 32],
        );
        let classification = backend.classify_from_validation(0.9, &hg);
        assert_eq!(classification.empirical, LocalEmpiricalLevel::Testimonial);
    }

    #[test]
    fn validation_classification_does_not_depend_on_low_connectivity() {
        let backend = SymthaeaBackend::new();
        let hg = HyperGradient::new(
            "node-2".to_string(),
            1,
            vec![64u8; HV16_BYTES],
            0.1,
            4_000_000,
            1.0,
            [0u8; 32],
        );
        let _classification = backend.classify_from_validation(0.05, &hg);
    }

    #[test]
    fn validation_classification_is_deterministic_for_same_inputs() {
        let backend = SymthaeaBackend::new();
        let hg = HyperGradient::new(
            "node-3".to_string(),
            1,
            vec![200u8; HV16_BYTES],
            0.7,
            4_000_000,
            5.0,
            [1u8; 32],
        );
        let c1 = backend.classify_from_validation(0.5, &hg);
        let c2 = backend.classify_from_validation(0.5, &hg);
        assert_eq!(format!("{:?}", c1), format!("{:?}", c2));
    }

    // ============================================================================
    // ConsciousnessAttestationData tests
    // ============================================================================

    #[test]
    fn test_create_consciousness_attestation_data_from_quality() {
        let cvec = ConsciousnessVector {
            spectral_connectivity: Some(0.7),
            true_phi: None,
            phi_fast: Some(0.8),
            entropy: Some(0.6),
            coherence: Some(0.9),
            epistemic_confidence: Some(0.85),
        };
        let expected_composite = cvec.composite();
        let quality = QualityScore {
            accuracy: 0.9,
            loss: 0.1,
            spectral: SpectralConnectivityAssessment::new(0.4, 0.7),
            consciousness_vector: cvec,
            epistemic_confidence: 0.85,
            is_anomalous: false,
            similarity: None,
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::None,
            causes: Vec::new(),
        };

        let attestation =
            create_consciousness_attestation_data(&quality, "did:key:z6MkTest", 42).unwrap();

        assert_eq!(attestation.agent_did, "did:key:z6MkTest");
        assert!(
            (attestation.consciousness_level - expected_composite).abs() < 1e-6,
            "should be composite of consciousness_vector"
        );
        assert_eq!(attestation.cycle_id, 42);
        assert_eq!(attestation.source, "symthaea");
        assert!(
            attestation.signature.is_empty(),
            "signature should be empty until caller signs"
        );
        assert!(
            attestation.captured_at_us > 0,
            "timestamp should be populated"
        );
    }

    #[test]
    fn test_attestation_rejects_out_of_range_consciousness() {
        let cvec = ConsciousnessVector {
            spectral_connectivity: Some(1.5),
            true_phi: Some(1.5),
            phi_fast: None,
            entropy: Some(1.5),
            coherence: Some(1.5),
            epistemic_confidence: Some(1.5),
        };
        let quality = QualityScore {
            accuracy: 0.9,
            loss: 0.1,
            spectral: SpectralConnectivityAssessment::new(0.0, 1.5),
            consciousness_vector: cvec,
            epistemic_confidence: 0.85,
            is_anomalous: false,
            similarity: None,
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::None,
            causes: Vec::new(),
        };

        let result = create_consciousness_attestation_data(&quality, "did:key:z6MkTest", 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_attestation_sign_message_is_deterministic() {
        let mut attestation = ConsciousnessAttestationData {
            agent_did: "did:key:z6MkABC".to_string(),
            consciousness_level: 0.654321,
            cycle_id: 100,
            captured_at_us: 1708000000_000000,
            signature: Vec::new(),
            source: "symthaea".to_string(),
            consciousness_vector: None,
        };

        let msg1 = attestation.sign_message().unwrap();
        let msg2 = attestation.sign_message().unwrap();
        assert_eq!(msg1, msg2, "Sign message should be deterministic");

        assert!(msg1.starts_with(b"symthaea-consciousness-attestation:v2\0"));

        // Changing consciousness_level changes the message
        attestation.consciousness_level = 0.999;
        let msg3 = attestation.sign_message().unwrap();
        assert_ne!(msg2, msg3);

        // Source and vector fields are governance-relevant and must also be covered.
        attestation.consciousness_level = 0.654321;
        attestation.source = "different-source".to_string();
        assert_ne!(msg2, attestation.sign_message().unwrap());

        attestation.source = "symthaea".to_string();
        attestation.consciousness_vector = Some(ConsciousnessVectorSerde {
            spectral_connectivity: Some(0.5),
            true_phi: None,
            phi_fast: None,
            entropy: None,
            coherence: None,
            epistemic_confidence: None,
        });
        assert_ne!(msg2, attestation.sign_message().unwrap());
    }

    #[test]
    fn test_attestation_data_serializable() {
        let attestation = ConsciousnessAttestationData {
            agent_did: "did:key:z6MkTest".to_string(),
            consciousness_level: 0.5,
            cycle_id: 1,
            captured_at_us: 1708000000_000000,
            signature: vec![1, 2, 3],
            source: "symthaea".to_string(),
            consciousness_vector: None,
        };

        let json = serde_json::to_string(&attestation).unwrap();
        let decoded: ConsciousnessAttestationData = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent_did, attestation.agent_did);
        assert!((decoded.consciousness_level - attestation.consciousness_level).abs() < 1e-10);
        assert_eq!(decoded.signature, vec![1, 2, 3]);
    }

    // ============================================================================
    // Original PoGQ tests
    // ============================================================================

    #[test]
    fn test_pogq_from_quality_score_mapping_behaves_sensibly() {
        // High-confidence, non-anomalous update should yield high PoGQ quality
        // and relatively low entropy.
        let high = QualityScore {
            accuracy: 0.9,
            loss: 0.1,
            spectral: SpectralConnectivityAssessment::new(0.4, 0.6),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.95,
            is_anomalous: false,
            similarity: Some(0.9),
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::None,
            causes: Vec::new(),
        };

        let pogq_high = pogq_from_quality_score(&high);
        assert!(
            pogq_high.quality > 0.8,
            "expected high PoGQ quality from high confidence"
        );
        assert!(
            pogq_high.entropy < 0.3,
            "expected relatively low entropy for non-anomalous update"
        );

        // Strongly anomalous update (large Φ drop and low confidence) should
        // have low quality and high entropy.
        let bad = QualityScore {
            accuracy: 0.4,
            loss: 1.5,
            spectral: SpectralConnectivityAssessment::new(0.6, 0.2),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: 0.1,
            is_anomalous: true,
            similarity: Some(0.2),
            is_ambiguous: true,
            severity: ConsciousAnomalySeverity::Severe,
            causes: vec!["integration_drop".into(), "low_confidence".into()],
        };

        let pogq_bad = pogq_from_quality_score(&bad);
        assert!(
            pogq_bad.quality < 0.5,
            "expected low PoGQ quality from low confidence"
        );
        assert!(
            pogq_bad.entropy > pogq_high.entropy,
            "expected higher entropy for anomalous update"
        );
    }

    #[test]
    fn malformed_quality_maps_to_fail_closed_pogq() {
        let malformed = QualityScore {
            accuracy: 0.9,
            loss: 0.1,
            spectral: SpectralConnectivityAssessment::new(0.4, 0.6),
            consciousness_vector: ConsciousnessVector::default(),
            epistemic_confidence: f32::NAN,
            is_anomalous: false,
            similarity: Some(0.9),
            is_ambiguous: false,
            severity: ConsciousAnomalySeverity::None,
            causes: Vec::new(),
        };

        let pogq = pogq_from_quality_score(&malformed);
        assert_eq!(pogq.quality, 0.0);
        assert_eq!(pogq.consistency, 0.0);
        assert_eq!(pogq.entropy, 1.0);
    }

    // ============================================================================
    // ConsciousnessVector tests
    // ============================================================================

    #[test]
    fn cvector_composite_all_populated() {
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(0.5),
            true_phi: Some(0.8),
            phi_fast: Some(0.7), // ignored because true_phi is present
            entropy: Some(0.6),
            coherence: Some(0.9),
            epistemic_confidence: Some(0.75),
        };
        let c = cv.composite();
        // Weights: phi(true)=0.35, coherence=0.20, entropy=0.15, epistemic=0.15, spectral=0.15
        let expected = 0.35 * 0.8 + 0.20 * 0.9 + 0.15 * 0.6 + 0.15 * 0.75 + 0.15 * 0.5;
        assert!((c - expected).abs() < 1e-10, "c={c}, expected={expected}");
    }

    #[test]
    fn cvector_composite_partial_population() {
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(0.5),
            true_phi: None,
            phi_fast: None,
            entropy: None,
            coherence: Some(0.8),
            epistemic_confidence: None,
        };
        let c = cv.composite();
        // Only spectral (0.15) + coherence (0.20) populated
        let expected = (0.15 * 0.5 + 0.20 * 0.8) / (0.15 + 0.20);
        assert!((c - expected).abs() < 1e-10, "c={c}, expected={expected}");
    }

    #[test]
    fn cvector_composite_empty_returns_zero() {
        let cv = ConsciousnessVector::default();
        assert_eq!(cv.composite(), 0.0);
    }

    #[test]
    fn cvector_best_phi_fallback_chain() {
        // true_phi > phi_fast > spectral_connectivity
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(0.3),
            true_phi: Some(0.9),
            phi_fast: Some(0.7),
            ..ConsciousnessVector::default()
        };
        assert!(
            (cv.best_phi() - 0.9).abs() < 1e-10,
            "should prefer true_phi"
        );

        let cv2 = ConsciousnessVector {
            spectral_connectivity: Some(0.3),
            true_phi: None,
            phi_fast: Some(0.7),
            ..ConsciousnessVector::default()
        };
        assert!(
            (cv2.best_phi() - 0.7).abs() < 1e-10,
            "should fall back to phi_fast"
        );

        let cv3 = ConsciousnessVector {
            spectral_connectivity: Some(0.3),
            true_phi: None,
            phi_fast: None,
            ..ConsciousnessVector::default()
        };
        assert!(
            (cv3.best_phi() - 0.3).abs() < 1e-10,
            "should fall back to spectral"
        );

        let cv4 = ConsciousnessVector::default();
        assert_eq!(
            cv4.best_phi(),
            0.0,
            "should return 0 when nothing populated"
        );
    }

    #[test]
    fn cvector_populated_count() {
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(0.5),
            true_phi: None,
            phi_fast: Some(0.7),
            entropy: Some(0.6),
            coherence: None,
            epistemic_confidence: None,
        };
        assert_eq!(cv.populated_count(), 3);
        assert_eq!(ConsciousnessVector::default().populated_count(), 0);
    }

    #[test]
    fn cvector_composite_clamped() {
        // Values > 1.0 should still produce composite <= 1.0
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(2.0),
            true_phi: Some(2.0),
            phi_fast: None,
            entropy: Some(2.0),
            coherence: Some(2.0),
            epistemic_confidence: Some(2.0),
        };
        assert_eq!(cv.composite(), 1.0, "composite should clamp to 1.0");
    }

    #[test]
    fn cvector_serialization_roundtrip() {
        let cv = ConsciousnessVector {
            spectral_connectivity: Some(0.5),
            true_phi: Some(0.8),
            phi_fast: None,
            entropy: Some(0.6),
            coherence: Some(0.9),
            epistemic_confidence: Some(0.75),
        };
        let json = serde_json::to_string(&cv).unwrap();
        let decoded: ConsciousnessVector = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.spectral_connectivity, cv.spectral_connectivity);
        assert_eq!(decoded.true_phi, cv.true_phi);
        assert_eq!(decoded.phi_fast, cv.phi_fast);
        assert!((decoded.composite() - cv.composite()).abs() < 1e-10);
    }
}
