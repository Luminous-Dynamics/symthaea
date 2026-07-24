// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-aesthetic
//!
//! Auditable, modality-independent aesthetic evidence and feedback for Symthaea's
//! creative systems.
//!
//! The crate does not treat one scalar as a universal definition of beauty. It
//! keeps artifact measurements, context alignment, novelty, learned preference,
//! policy utility, and uncertainty separate until an explicit policy combines them.
//!
//! # Architecture
//!
//! ```text
//! Artifact -> modality analyzer -> ExtractionReport -> ArtifactEvidence
//! ArtifactEvidence + context + novelty + preference -> AestheticAssessment
//! AestheticAssessment -> explanation / release evidence / bounded feedback
//! PreferenceStudyLedger -> held-out calibration -> evidence manifest + release gate
//! Registered extractor + report + assessment -> evaluation envelope -> operational gate
//! API + schemas + registry + extractor -> contract snapshot
//! Pipeline output + contract snapshot -> portable receipt + self-verifying archive
//! Archive + audit + benchmark + integration profile -> adoption certification
//! Telemetry -> drift / robustness / replay / SLO evidence -> accountable deployment
//! ```
//!
//! The legacy [`AestheticEvaluator`] and [`AestheticScore`] APIs remain available.
//! New integrations should prefer [`prelude`], [`EvidenceExtractor`],
//! [`ExtractionReport`], and [`AestheticAssessment`] so measurement support,
//! compatibility, and policy provenance remain visible.
//!
//! # Theoretical priors
//!
//! Birkhoff, Shannon, golden-ratio, and Berlyne-inspired metrics are exposed as
//! testable priors. They are not represented as universally validated laws of
//! human aesthetic preference.

#![deny(unsafe_code)]

pub mod birkhoff;
pub mod diagnostics;
pub mod feedback;
pub mod golden;
pub mod harmony;
pub mod information;
pub mod novelty;
pub mod session;
pub mod synesthesia;
pub mod valence_arousal;

pub use harmony::{HarmonyEvidenceLedger, HarmonyEvidenceSource};
pub use valence_arousal::{MusicalParams, ValenceArousal, from_core_affect, lerp_va};

use serde::{Deserialize, Serialize};

// ─── Core Trait ───────────────────────────────────────────────────────────────

/// Modality-independent aesthetic evaluation.
///
/// Implement this for each creative modality (visual, musical, poetic).
/// The evaluator produces an `AestheticScore` from an artifact of type `A`.
pub trait AestheticEvaluator<A> {
    /// Evaluate the aesthetic quality of an artifact.
    ///
    /// Returns a score with all dimensions in [0.0, 1.0].
    fn evaluate(&self, artifact: &A) -> AestheticScore;
}

// ─── Core Types ──────────────────────────────────────────────────────────────

/// Multi-dimensional aesthetic score.
///
/// Each dimension captures a different aspect of beauty, allowing the creative
/// system to optimize for specific aesthetic qualities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AestheticScore {
    /// Symmetry, balance, repetition, structural regularity.
    /// High order = pleasing regularity (Birkhoff's "O").
    pub order: f32,

    /// Information content, variety, richness.
    /// High complexity = more elements to process (Birkhoff's "C").
    pub complexity: f32,

    /// Deviation from aesthetic expectation (the EMA of recent scores).
    /// High surprise = novel aesthetic territory.
    pub surprise: f32,

    /// Consonance with the Eight Harmonies projection.
    /// High harmony = alignment with the system's value basis.
    pub harmony: f32,

    /// Birkhoff's aesthetic measure: order / complexity.
    /// The classical beauty metric.
    pub birkhoff: f32,

    /// Novelty-aware utility: 80% intrinsic evidence and 20% surprise.
    /// Use [`Self::intrinsic_composite`] when updating quality expectations.
    pub composite: f32,
}

impl AestheticScore {
    /// Create a score with all dimensions set to the same value.
    pub fn uniform(value: f32) -> Self {
        let v = sanitize_unit(value);
        Self {
            order: v,
            complexity: v,
            surprise: v,
            harmony: v,
            birkhoff: v,
            composite: v,
        }
    }

    /// Create a zero score (no aesthetic value).
    pub fn zero() -> Self {
        Self::uniform(0.0)
    }

    /// Berlyne-style preference for moderate complexity.
    pub fn complexity_balance(&self) -> f32 {
        let complexity = sanitize_unit(self.complexity);
        (1.0 - (complexity - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    }

    /// Score the artifact's intrinsic evidence without novelty or history.
    ///
    /// The weights are the legacy non-surprise weights renormalized to sum to
    /// one. This creates a stable expectation target: a work cannot raise its
    /// own baseline merely by being unexpected.
    pub fn intrinsic_composite(&self) -> f32 {
        (0.375 * sanitize_unit(self.birkhoff)
            + 0.3125 * sanitize_unit(self.harmony)
            + 0.1875 * sanitize_unit(self.order)
            + 0.125 * self.complexity_balance())
        .clamp(0.0, 1.0)
    }

    /// Absolute prediction error against an intrinsic-score expectation.
    pub fn surprise_against(&self, expectation: f32) -> f32 {
        (self.intrinsic_composite() - sanitize_unit(expectation))
            .abs()
            .clamp(0.0, 1.0)
    }

    /// Return a copy with a resolved surprise term and recomputed composite.
    pub fn with_surprise(mut self, surprise: f32) -> Self {
        self.surprise = sanitize_unit(surprise);
        self.compute_composite();
        self
    }

    /// Compute the novelty-aware utility score from intrinsic evidence and a
    /// separately supplied surprise term.
    ///
    /// Intrinsic evidence contributes 80%; surprise contributes 20%. Keeping
    /// the intrinsic score available separately prevents circular expectation
    /// updates while retaining the original public weighting.
    pub fn compute_composite(&mut self) {
        self.order = sanitize_unit(self.order);
        self.complexity = sanitize_unit(self.complexity);
        self.surprise = sanitize_unit(self.surprise);
        self.harmony = sanitize_unit(self.harmony);
        self.birkhoff = sanitize_unit(self.birkhoff);
        self.composite = (0.80 * self.intrinsic_composite() + 0.20 * self.surprise).clamp(0.0, 1.0);
    }
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_signed(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

impl Default for AestheticScore {
    fn default() -> Self {
        Self::zero()
    }
}

/// Aesthetic feedback signal for the cognitive loop.
///
/// Converts an `AestheticScore` into neuromodulator deltas that can be injected
/// into the `NeuromodulatorBath`. This is the key loop-closing mechanism:
/// beautiful outputs reinforce creative exploration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AestheticFeedback {
    /// Dopamine delta: reward for exceeding the aesthetic EMA.
    /// Positive when current score > running average (reward prediction error).
    pub dopamine_delta: f32,

    /// Serotonin delta: satisfaction from harmonic alignment.
    /// Proportional to the harmony dimension of the score.
    pub serotonin_delta: f32,

    /// Noradrenaline signal: surprise/novelty for the exploration system.
    /// High when aesthetic surprise is high.
    pub surprise_signal: f32,

    /// Projection of the artifact's aesthetic character onto the Eight Harmonies.
    /// Used by the creative loop to bias future generation toward resonant harmonies.
    pub harmony_projection: [f32; 8],
}

impl AestheticFeedback {
    /// No-feedback signal (all zeros).
    pub fn neutral() -> Self {
        Self {
            dopamine_delta: 0.0,
            serotonin_delta: 0.0,
            surprise_signal: 0.0,
            harmony_projection: [0.0; 8],
        }
    }
}

impl Default for AestheticFeedback {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Configuration for the aesthetic evaluation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticConfig {
    /// EMA alpha for tracking the aesthetic running average.
    /// Lower = smoother, higher = more responsive.
    pub ema_alpha: f32,

    /// Dopamine reward scaling factor.
    pub dopamine_scale: f32,

    /// Serotonin satisfaction scaling factor.
    pub serotonin_scale: f32,

    /// Surprise signal scaling factor.
    pub surprise_scale: f32,

    /// Minimum score delta (vs EMA) to trigger dopamine reward.
    pub reward_threshold: f32,
}

/// Validation failure for [`AestheticConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AestheticConfigError {
    NonFinite(&'static str),
    OutOfRange(&'static str),
}

impl std::fmt::Display for AestheticConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
            Self::OutOfRange(field) => write!(f, "{field} is outside its supported range"),
        }
    }
}

impl std::error::Error for AestheticConfigError {}

impl AestheticConfig {
    /// Validate feedback dynamics before constructing a tracker.
    pub fn validate(&self) -> Result<(), AestheticConfigError> {
        for (name, value) in [
            ("ema_alpha", self.ema_alpha),
            ("dopamine_scale", self.dopamine_scale),
            ("serotonin_scale", self.serotonin_scale),
            ("surprise_scale", self.surprise_scale),
            ("reward_threshold", self.reward_threshold),
        ] {
            if !value.is_finite() {
                return Err(AestheticConfigError::NonFinite(name));
            }
        }
        if !(0.0..=1.0).contains(&self.ema_alpha) {
            return Err(AestheticConfigError::OutOfRange("ema_alpha"));
        }
        if self.dopamine_scale < 0.0 {
            return Err(AestheticConfigError::OutOfRange("dopamine_scale"));
        }
        if self.serotonin_scale < 0.0 {
            return Err(AestheticConfigError::OutOfRange("serotonin_scale"));
        }
        if self.surprise_scale < 0.0 {
            return Err(AestheticConfigError::OutOfRange("surprise_scale"));
        }
        if !(0.0..=1.0).contains(&self.reward_threshold) {
            return Err(AestheticConfigError::OutOfRange("reward_threshold"));
        }
        Ok(())
    }

    /// Fail-safe normalization for legacy callers that use [`AestheticTracker::new`].
    pub fn sanitized(mut self) -> Self {
        let defaults = Self::default();
        if !self.ema_alpha.is_finite() {
            self.ema_alpha = defaults.ema_alpha;
        }
        if !self.dopamine_scale.is_finite() {
            self.dopamine_scale = defaults.dopamine_scale;
        }
        if !self.serotonin_scale.is_finite() {
            self.serotonin_scale = defaults.serotonin_scale;
        }
        if !self.surprise_scale.is_finite() {
            self.surprise_scale = defaults.surprise_scale;
        }
        if !self.reward_threshold.is_finite() {
            self.reward_threshold = defaults.reward_threshold;
        }
        self.ema_alpha = self.ema_alpha.clamp(0.0, 1.0);
        self.dopamine_scale = self.dopamine_scale.max(0.0);
        self.serotonin_scale = self.serotonin_scale.max(0.0);
        self.surprise_scale = self.surprise_scale.max(0.0);
        self.reward_threshold = self.reward_threshold.clamp(0.0, 1.0);
        self
    }
}

impl Default for AestheticConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.15,
            dopamine_scale: 0.10,
            serotonin_scale: 0.05,
            surprise_scale: 0.10,
            reward_threshold: 0.02,
        }
    }
}

/// Persisted aesthetic memory — survives across sessions.
///
/// Stores the intrinsic-quality expectation and accumulated harmony bias so
/// Symthaea can carry an aesthetic identity across process boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticMemory {
    /// Persistence schema version. Version zero is accepted as the legacy
    /// unversioned representation and upgraded on the next save.
    #[serde(default)]
    pub schema_version: u32,
    /// EMA of intrinsic scores accumulated across all sessions.
    pub ema: f32,
    /// Accumulated compatibility view of harmony preference.
    pub harmony_bias: [f32; 8],
    /// Contrastive evidence supporting the compatibility bias.
    #[serde(default)]
    pub harmony_evidence: HarmonyEvidenceLedger,
    /// Total evaluations ever performed.
    pub total_evaluations: u64,
    /// Sessions completed.
    pub session_count: u32,
}

pub const AESTHETIC_MEMORY_SCHEMA_VERSION: u32 = 2;

/// Failure while loading, validating, or atomically saving aesthetic memory.
#[derive(Debug)]
pub enum AestheticMemoryError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedSchema(u32),
    InvalidState(&'static str),
}

impl std::fmt::Display for AestheticMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "aesthetic memory I/O failed: {error}"),
            Self::Json(error) => write!(f, "aesthetic memory JSON failed: {error}"),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported aesthetic memory schema version {version}")
            }
            Self::InvalidState(field) => write!(f, "invalid aesthetic memory field: {field}"),
        }
    }
}

impl std::error::Error for AestheticMemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedSchema(_) | Self::InvalidState(_) => None,
        }
    }
}

impl From<std::io::Error> for AestheticMemoryError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for AestheticMemoryError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl AestheticMemory {
    pub fn new() -> Self {
        Self {
            schema_version: AESTHETIC_MEMORY_SCHEMA_VERSION,
            ema: 0.5,
            harmony_bias: [0.0; 8],
            harmony_evidence: HarmonyEvidenceLedger::new(),
            total_evaluations: 0,
            session_count: 0,
        }
    }

    /// Validate persisted state before it can influence feedback dynamics.
    pub fn validate(&self) -> Result<(), AestheticMemoryError> {
        if self.schema_version > AESTHETIC_MEMORY_SCHEMA_VERSION {
            return Err(AestheticMemoryError::UnsupportedSchema(self.schema_version));
        }
        if !self.ema.is_finite() || !(0.0..=1.0).contains(&self.ema) {
            return Err(AestheticMemoryError::InvalidState("ema"));
        }
        if self
            .harmony_bias
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(AestheticMemoryError::InvalidState("harmony_bias"));
        }
        if !self.harmony_evidence.validate() {
            return Err(AestheticMemoryError::InvalidState("harmony_evidence"));
        }
        Ok(())
    }

    /// Load and validate memory, preserving the cause of any failure.
    pub fn try_load(path: &std::path::Path) -> Result<Self, AestheticMemoryError> {
        let json = std::fs::read_to_string(path)?;
        let mut memory: Self = serde_json::from_str(&json)?;
        memory.validate()?;
        if memory.schema_version < AESTHETIC_MEMORY_SCHEMA_VERSION {
            memory.schema_version = AESTHETIC_MEMORY_SCHEMA_VERSION;
        }
        Ok(memory)
    }

    /// Compatibility loader that falls back to a fresh identity.
    ///
    /// New code should prefer [`Self::try_load`] so corruption and permission
    /// failures remain observable.
    pub fn load(path: &std::path::Path) -> Self {
        Self::try_load(path).unwrap_or_default()
    }

    /// Atomically persist validated memory using write, sync, then rename.
    pub fn try_save(&self, path: &std::path::Path) -> Result<(), AestheticMemoryError> {
        use std::io::Write;

        self.validate()?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("aesthetic-memory.json");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary =
            path.with_file_name(format!(".{file_name}.{}.{nonce}.tmp", std::process::id()));
        let json = serde_json::to_vec_pretty(self)?;

        let result = (|| -> Result<(), AestheticMemoryError> {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&temporary)?;
            file.write_all(&json)?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }

    /// Compatibility saver. New code should prefer [`Self::try_save`].
    pub fn save(&self, path: &std::path::Path) {
        let _ = self.try_save(path);
    }
}

impl Default for AestheticMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful aesthetic tracker that maintains a running EMA of scores
/// and produces feedback signals relative to that expectation.
#[derive(Debug, Clone)]
pub struct AestheticTracker {
    config: AestheticConfig,
    /// Exponential moving average of intrinsic quality scores.
    ema: f32,
    /// Compatibility view of contrastively learned harmony preference.
    harmony_bias: [f32; 8],
    /// Active-versus-inactive evidence, separated by provenance.
    harmony_evidence: HarmonyEvidenceLedger,
    /// Evaluations performed in the current session.
    evaluation_count: u64,
    /// Evaluations performed over the persisted lifetime, including this session.
    lifetime_evaluation_count: u64,
}

impl AestheticTracker {
    pub fn new(config: AestheticConfig) -> Self {
        Self {
            config: config.sanitized(),
            ema: 0.5, // neutral starting expectation
            harmony_bias: [0.0; 8],
            harmony_evidence: HarmonyEvidenceLedger::new(),
            evaluation_count: 0,
            lifetime_evaluation_count: 0,
        }
    }

    /// Construct a tracker only if the configuration is valid.
    pub fn try_new(config: AestheticConfig) -> Result<Self, AestheticConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            ema: 0.5,
            harmony_bias: [0.0; 8],
            harmony_evidence: HarmonyEvidenceLedger::new(),
            evaluation_count: 0,
            lifetime_evaluation_count: 0,
        })
    }

    /// Create a tracker pre-warmed from persisted memory.
    ///
    /// The EMA and harmony bias from previous sessions seed this session,
    /// so Symthaea's aesthetic expectations carry forward over time.
    pub fn from_memory(config: AestheticConfig, memory: &AestheticMemory) -> Self {
        Self {
            config: config.sanitized(),
            ema: sanitize_unit(memory.ema),
            harmony_bias: std::array::from_fn(|i| sanitize_unit(memory.harmony_bias[i])),
            harmony_evidence: memory.harmony_evidence.clone(),
            evaluation_count: 0,
            lifetime_evaluation_count: memory.total_evaluations,
        }
    }

    /// Snapshot current state into an `AestheticMemory` for persistence.
    pub fn to_memory(&self, previous: &AestheticMemory) -> AestheticMemory {
        AestheticMemory {
            schema_version: AESTHETIC_MEMORY_SCHEMA_VERSION,
            ema: self.ema,
            harmony_bias: self.harmony_bias,
            harmony_evidence: self.harmony_evidence.clone(),
            total_evaluations: self.lifetime_evaluation_count,
            session_count: previous.session_count.saturating_add(1),
        }
    }

    /// The accumulated harmony bias: which harmonies have historically scored well.
    /// Values in [0, 1] — higher means this harmony correlates with beautiful output.
    pub fn harmony_bias(&self) -> &[f32; 8] {
        &self.harmony_bias
    }

    /// Process an aesthetic score and produce feedback.
    ///
    /// Updates the EMA and computes neuromodulator deltas relative to expectation.
    pub fn process(
        &mut self,
        score: &AestheticScore,
        harmony_activations: &[f32; 8],
    ) -> AestheticFeedback {
        self.evaluation_count = self.evaluation_count.saturating_add(1);
        self.lifetime_evaluation_count = self.lifetime_evaluation_count.saturating_add(1);

        // Resolve surprise from intrinsic prediction error. The expectation is
        // updated from intrinsic quality only, preventing surprise from
        // recursively inflating its own future baseline.
        let intrinsic = score.intrinsic_composite();
        let surprise = score.surprise_against(self.ema);
        let delta = intrinsic - self.ema;

        // Update the intrinsic-quality expectation.
        self.ema = self.ema * (1.0 - self.config.ema_alpha) + intrinsic * self.config.ema_alpha;

        // Dopamine: reward prediction error (positive when exceeding expectation)
        let dopamine_delta = if delta > self.config.reward_threshold {
            (delta * self.config.dopamine_scale).min(0.15)
        } else if delta < -self.config.reward_threshold {
            // Mild negative signal for disappointing output (not as strong as reward)
            (delta * self.config.dopamine_scale * 0.5).max(-0.05)
        } else {
            0.0
        };

        // Serotonin: proportional to harmony alignment
        let serotonin_delta = sanitize_unit(score.harmony) * self.config.serotonin_scale;

        // Surprise signal for exploration system
        let surprise_signal = surprise * self.config.surprise_scale;

        // Project aesthetic quality onto harmonies using the system's current activations.
        // Harmonies that are active AND the artwork scored well get reinforced.
        let harmony_projection: [f32; 8] =
            std::array::from_fn(|i| sanitize_unit(harmony_activations[i]) * intrinsic);

        // Learn contrastively rather than rewarding whichever harmonies happen
        // to be most prevalent. Self-evaluation is intentionally low-confidence.
        self.harmony_evidence.observe(
            HarmonyEvidenceSource::SelfEvaluation,
            harmony_activations,
            intrinsic,
            0.25,
        );
        self.refresh_harmony_bias();

        AestheticFeedback {
            dopamine_delta,
            serotonin_delta,
            surprise_signal,
            harmony_projection,
        }
    }

    fn refresh_harmony_bias(&mut self) {
        for index in 0..8 {
            self.harmony_bias[index] = self
                .harmony_evidence
                .preference(index, self.harmony_bias[index]);
        }
    }

    /// Current EMA value (the system's aesthetic expectation).
    pub fn expectation(&self) -> f32 {
        self.ema
    }

    /// Number of evaluations performed in the current session.
    pub fn evaluation_count(&self) -> u64 {
        self.evaluation_count
    }

    /// Total number of evaluations represented by this tracker.
    pub fn total_evaluation_count(&self) -> u64 {
        self.lifetime_evaluation_count
    }

    /// Reset session-local expectation and count while preserving lifetime taste.
    pub fn reset(&mut self) {
        self.ema = 0.5;
        self.evaluation_count = 0;
        // harmony_bias intentionally preserved — it's long-term taste, not session state
    }

    // ── Human Feedback API ───────────────────────────────────────────────────

    /// Incorporate human feedback into the aesthetic system.
    ///
    /// `rating` is a scalar from -1.0 (terrible) through 0.0 (neutral) to +1.0 (beautiful).
    /// `harmony_activations` is the harmony state at the time the rated piece was generated.
    ///
    /// Human feedback carries 10x more weight than self-evaluation because humans
    /// are the ground truth for aesthetic quality. The system recalibrates its
    /// EMA and harmony bias toward the human's judgement.
    ///
    /// # Example
    ///
    /// ```
    /// # use symthaea_aesthetic::{AestheticTracker, AestheticConfig};
    /// let mut tracker = AestheticTracker::new(AestheticConfig::default());
    /// let harmonies = [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2];
    /// tracker.human_feedback(0.8, &harmonies); // "that was beautiful"
    /// ```
    pub fn human_feedback(
        &mut self,
        rating: f32,
        harmony_activations: &[f32; 8],
    ) -> AestheticFeedback {
        let rating = sanitize_signed(rating);
        // Map rating [-1, 1] to score [0, 1]
        let score = (rating + 1.0) * 0.5;

        self.evaluation_count = self.evaluation_count.saturating_add(1);
        self.lifetime_evaluation_count = self.lifetime_evaluation_count.saturating_add(1);

        let delta = score - self.ema;

        // Human feedback EMA alpha is 10x stronger than self-evaluation.
        // One "beautiful" from a human shifts expectations more than 10 self-scores.
        let human_alpha = (self.config.ema_alpha * 10.0).min(0.5);
        self.ema = self.ema * (1.0 - human_alpha) + score * human_alpha;

        // Keep human evidence separate and give it full confidence. The ledger
        // will only infer a preference once active and inactive contrasts exist.
        self.harmony_evidence.observe(
            HarmonyEvidenceSource::Human,
            harmony_activations,
            score,
            1.0,
        );
        self.refresh_harmony_bias();

        // Strong dopamine signal: human approval is the ultimate reward
        let dopamine_delta = delta * self.config.dopamine_scale * 2.0;
        let serotonin_delta = if rating > 0.3 {
            rating * self.config.serotonin_scale * 1.5 // warm glow from approval
        } else if rating < -0.3 {
            rating * self.config.serotonin_scale * 0.5 // mild serotonin dip
        } else {
            0.0
        };

        let harmony_projection: [f32; 8] =
            std::array::from_fn(|i| sanitize_unit(harmony_activations[i]) * score);

        AestheticFeedback {
            dopamine_delta: dopamine_delta.clamp(-0.2, 0.3),
            serotonin_delta: serotonin_delta.clamp(-0.1, 0.2),
            surprise_signal: delta.abs() * self.config.surprise_scale,
            harmony_projection,
        }
    }

    /// Batch human feedback: apply multiple ratings at once.
    ///
    /// Useful for catching up on feedback from a listening session.
    pub fn human_feedback_batch(&mut self, ratings: &[(f32, [f32; 8])]) -> Vec<AestheticFeedback> {
        ratings
            .iter()
            .map(|(rating, harmonies)| self.human_feedback(*rating, harmonies))
            .collect()
    }

    /// Human feedback when the generation-time harmony state is unknown
    /// (e.g. Symthaea's facade art path, which deliberately does not
    /// fabricate harmony readings it cannot observe).
    ///
    /// Applies the same 10×-weight EMA recalibration and reward signals as
    /// [`Self::human_feedback`] but leaves the harmony bias untouched —
    /// absent information should mean *no* bias update, not a decay toward
    /// zero (which is what passing an all-zero activation array would do).
    pub fn human_feedback_unattributed(&mut self, rating: f32) -> AestheticFeedback {
        let rating = sanitize_signed(rating);
        let score = (rating + 1.0) * 0.5;

        self.evaluation_count = self.evaluation_count.saturating_add(1);
        self.lifetime_evaluation_count = self.lifetime_evaluation_count.saturating_add(1);

        let delta = score - self.ema;
        let human_alpha = (self.config.ema_alpha * 10.0).min(0.5);
        self.ema = self.ema * (1.0 - human_alpha) + score * human_alpha;

        let dopamine_delta = delta * self.config.dopamine_scale * 2.0;
        let serotonin_delta = if rating > 0.3 {
            rating * self.config.serotonin_scale * 1.5
        } else if rating < -0.3 {
            rating * self.config.serotonin_scale * 0.5
        } else {
            0.0
        };

        AestheticFeedback {
            dopamine_delta: dopamine_delta.clamp(-0.2, 0.3),
            serotonin_delta: serotonin_delta.clamp(-0.1, 0.2),
            surprise_signal: delta.abs() * self.config.surprise_scale,
            harmony_projection: [0.0; 8],
        }
    }
}

impl Default for AestheticTracker {
    fn default() -> Self {
        Self::new(AestheticConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unattributed_feedback_moves_ema_but_not_bias() {
        let mut tracker = AestheticTracker::new(AestheticConfig::default());
        // Seed some real, attributed bias first.
        tracker.human_feedback(0.8, &[0.9; 8]);
        let bias_before = *tracker.harmony_bias();
        let ema_before = tracker.expectation();

        let fb = tracker.human_feedback_unattributed(-0.9);
        assert!(
            tracker.expectation() < ema_before,
            "negative rating must lower the EMA"
        );
        assert_eq!(
            bias_before,
            *tracker.harmony_bias(),
            "unattributed feedback must not touch harmony bias"
        );
        assert!(fb.dopamine_delta < 0.0, "disapproval is a negative reward");
    }

    #[test]
    fn score_composite_moderate_complexity_preferred() {
        let mut low = AestheticScore {
            order: 0.8,
            complexity: 0.1, // too simple
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        let mut mid = AestheticScore {
            order: 0.8,
            complexity: 0.5, // just right (Berlyne peak)
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        let mut high = AestheticScore {
            order: 0.8,
            complexity: 0.9, // too complex
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        low.compute_composite();
        mid.compute_composite();
        high.compute_composite();
        assert!(mid.composite > low.composite, "mid {mid:?} > low {low:?}");
        assert!(
            mid.composite > high.composite,
            "mid {mid:?} > high {high:?}"
        );
    }

    #[test]
    fn score_bounded() {
        let mut score = AestheticScore {
            order: 1.0,
            complexity: 0.5,
            surprise: 1.0,
            harmony: 1.0,
            birkhoff: 1.0,
            composite: 0.0,
        };
        score.compute_composite();
        assert!(score.composite >= 0.0 && score.composite <= 1.0);

        let mut zero_score = AestheticScore::zero();
        zero_score.compute_composite();
        assert!(zero_score.composite >= 0.0);
    }

    #[test]
    fn tracker_ema_updates() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        let score = AestheticScore::uniform(0.8);
        tracker.process(&score, &harmonies);

        // EMA should move toward 0.8 from initial 0.5
        assert!(tracker.expectation() > 0.5);
        assert!(tracker.expectation() < 0.8);
    }

    #[test]
    fn tracker_reward_on_exceeding_expectation() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        // First: set baseline low
        for _ in 0..10 {
            tracker.process(&AestheticScore::uniform(0.3), &harmonies);
        }
        let ema_before = tracker.expectation();

        // Now: provide a high score — should produce positive dopamine
        let high_score = AestheticScore::uniform(0.9);
        let feedback = tracker.process(&high_score, &harmonies);

        assert!(
            feedback.dopamine_delta > 0.0,
            "should reward exceeding EMA: delta={}",
            feedback.dopamine_delta
        );
        assert!(tracker.expectation() > ema_before, "EMA should increase");
    }

    #[test]
    fn tracker_disappointment_mild_negative() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        // Set baseline high
        for _ in 0..10 {
            tracker.process(&AestheticScore::uniform(0.9), &harmonies);
        }

        // Now: disappointing output
        let low_score = AestheticScore::uniform(0.2);
        let feedback = tracker.process(&low_score, &harmonies);

        assert!(
            feedback.dopamine_delta < 0.0,
            "should have mild negative dopamine: {}",
            feedback.dopamine_delta
        );
        // But not too negative (clamped)
        assert!(feedback.dopamine_delta >= -0.05);
    }

    #[test]
    fn feedback_neutral_is_zero() {
        let feedback = AestheticFeedback::neutral();
        assert_eq!(feedback.dopamine_delta, 0.0);
        assert_eq!(feedback.serotonin_delta, 0.0);
        assert_eq!(feedback.surprise_signal, 0.0);
        assert_eq!(feedback.harmony_projection, [0.0; 8]);
    }

    #[test]
    fn harmony_projection_scales_with_activation() {
        let mut tracker = AestheticTracker::default();
        let mut harmonies = [0.0f32; 8];
        harmonies[3] = 1.0; // Only InfinitePlay active

        let score = AestheticScore::uniform(0.8);
        let feedback = tracker.process(&score, &harmonies);

        assert!(feedback.harmony_projection[3] > 0.0);
        assert_eq!(feedback.harmony_projection[0], 0.0); // Inactive harmony
    }

    #[test]
    fn tracker_evaluation_count() {
        let mut tracker = AestheticTracker::default();
        assert_eq!(tracker.evaluation_count(), 0);

        tracker.process(&AestheticScore::uniform(0.5), &[0.5; 8]);
        assert_eq!(tracker.evaluation_count(), 1);

        tracker.process(&AestheticScore::uniform(0.5), &[0.5; 8]);
        assert_eq!(tracker.evaluation_count(), 2);
    }

    #[test]
    fn tracker_reset() {
        let mut tracker = AestheticTracker::default();
        tracker.process(&AestheticScore::uniform(0.9), &[0.5; 8]);
        tracker.reset();
        assert_eq!(tracker.expectation(), 0.5);
        assert_eq!(tracker.evaluation_count(), 0);
    }

    #[test]
    fn prediction_surprise_is_generated_when_score_field_is_zero() {
        let mut tracker = AestheticTracker::default();
        for _ in 0..12 {
            tracker.process(&AestheticScore::uniform(0.2), &[0.5; 8]);
        }
        let mut high = AestheticScore::uniform(0.9);
        high.surprise = 0.0;
        high.compute_composite();
        let feedback = tracker.process(&high, &[0.5; 8]);
        assert!(feedback.surprise_signal > 0.0);
    }

    #[test]
    fn intrinsic_expectation_is_not_novelty_recursive() {
        let mut tracker = AestheticTracker::default();
        let mut score = AestheticScore::uniform(0.6);
        score.surprise = 1.0;
        score.compute_composite();
        let intrinsic = score.intrinsic_composite();
        tracker.process(&score, &[0.5; 8]);
        let expected = 0.5 * (1.0 - AestheticConfig::default().ema_alpha)
            + intrinsic * AestheticConfig::default().ema_alpha;
        assert!((tracker.expectation() - expected).abs() < 1e-6);
    }

    #[test]
    fn invalid_config_is_rejected_or_sanitized() {
        let invalid = AestheticConfig {
            ema_alpha: 2.0,
            dopamine_scale: -1.0,
            serotonin_scale: f32::NAN,
            surprise_scale: 0.1,
            reward_threshold: 0.02,
        };
        assert!(AestheticTracker::try_new(invalid.clone()).is_err());
        let tracker = AestheticTracker::new(invalid);
        assert!(tracker.expectation().is_finite());
    }

    #[test]
    fn memory_roundtrip_is_validated_and_atomic() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-aesthetic-memory-{}.json",
            std::process::id()
        ));
        let memory = AestheticMemory {
            schema_version: AESTHETIC_MEMORY_SCHEMA_VERSION,
            ema: 0.7,
            harmony_bias: [0.2; 8],
            harmony_evidence: HarmonyEvidenceLedger::new(),
            total_evaluations: 42,
            session_count: 3,
        };
        memory.try_save(&path).expect("save memory");
        let loaded = AestheticMemory::try_load(&path).expect("load memory");
        assert_eq!(loaded.total_evaluations, 42);
        assert_eq!(loaded.session_count, 3);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn corrupt_memory_is_observable() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-aesthetic-corrupt-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, "{not-json").expect("write corrupt fixture");
        assert!(AestheticMemory::try_load(&path).is_err());
        assert_eq!(AestheticMemory::load(&path).total_evaluations, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn session_reset_does_not_erase_lifetime_count() {
        let memory = AestheticMemory {
            schema_version: AESTHETIC_MEMORY_SCHEMA_VERSION,
            ema: 0.5,
            harmony_bias: [0.0; 8],
            harmony_evidence: HarmonyEvidenceLedger::new(),
            total_evaluations: 100,
            session_count: 2,
        };
        let mut tracker = AestheticTracker::from_memory(AestheticConfig::default(), &memory);
        tracker.process(&AestheticScore::uniform(0.5), &[0.5; 8]);
        assert_eq!(tracker.evaluation_count(), 1);
        assert_eq!(tracker.total_evaluation_count(), 101);
        tracker.reset();
        assert_eq!(tracker.evaluation_count(), 0);
        assert_eq!(tracker.to_memory(&memory).total_evaluations, 101);
    }

    #[test]
    fn harmony_bias_requires_active_inactive_contrast() {
        let mut tracker = AestheticTracker::default();
        for _ in 0..20 {
            tracker.process(&AestheticScore::uniform(0.9), &[1.0; 8]);
        }
        assert_eq!(*tracker.harmony_bias(), [0.0; 8]);

        let mut active = [0.0; 8];
        active[0] = 1.0;
        for _ in 0..20 {
            tracker.human_feedback(0.9, &active);
            tracker.human_feedback(-0.9, &[0.0; 8]);
        }
        assert!(tracker.harmony_bias()[0] > 0.8);
    }

    #[test]
    fn non_finite_human_feedback_fails_safe() {
        let mut tracker = AestheticTracker::default();
        let feedback = tracker.human_feedback(f32::NAN, &[f32::NAN; 8]);
        assert!(tracker.expectation().is_finite());
        assert!(feedback.dopamine_delta.is_finite());
        assert!(feedback.serotonin_delta.is_finite());
        assert!(
            feedback
                .harmony_projection
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
