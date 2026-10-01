// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Ignorance Type Taxonomy
//!
//! Implements the 5-type ignorance taxonomy from GIS v1.0, extended in v4.0 with
//! Harmonic Ignorance for Eight Harmonies integration.
//!
//! | Symbol | Name | Description |
//! |--------|------|-------------|
//! | κ | Known | Information exists and is accessible |
//! | ι₁ | Known Unknown | We know we don't know this specific thing |
//! | ι₂ | Unknown Unknown | We don't know what we don't know |
//! | ι₃ | Impossible | Fundamentally unknowable |
//! | ι∞ | None | No ignorance detected under the active frame |
//!
//! ## GIS v4.0 Extension: Harmonic Ignorance
//!
//! Ignorance is not uniform across the Eight Harmonies. A gap in Care-Knowing
//! (Pan-Sentient Flourishing) differs fundamentally from a gap in Truth-Knowing
//! (Integral Wisdom).
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::{IgnoranceType, HarmonicIgnorance, Harmony};
//!
//! let mut harmonic_ig = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
//! harmonic_ig.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
//! harmonic_ig.add_affected_harmony(Harmony::IntegralWisdom, 0.6);
//!
//! println!("Primary gap: {:?}", harmonic_ig.primary_harmony_gap());
//! println!("Total impact: {:.2}", harmonic_ig.total_harmonic_impact());
//! ```

use std::time::SystemTime;

/// The 5-type ignorance taxonomy
///
/// Based on epistemological analysis of what kinds of "not knowing" exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IgnoranceType {
    /// No ignorance detected under the active evidence/model/schema/ontology frame (κ).
    None,

    /// Known - Information exists somewhere, we know what it is but don't have it (κ temporally limited)
    Known,

    /// Known Unknown (ι₁) - We know exactly what we don't know
    /// "I don't know the population of Tokyo, but I know the question is well-formed"
    KnownUnknown,

    /// Unknown Unknown (ι₂) - We don't know what we don't know
    /// "There may be relevant factors I'm not even aware of"
    Unknown,

    /// Impossible (ι₃) - Fundamentally unknowable
    /// "What is north of the North Pole?" - the question is malformed
    Impossible,
}

impl IgnoranceType {
    /// Can this type of ignorance be resolved in principle?
    pub fn is_resolvable(&self) -> bool {
        match self {
            Self::None => true,         // No ignorance detected under the active frame
            Self::Known => true,        // Just need to fetch it
            Self::KnownUnknown => true, // Can research it
            Self::Unknown => false,     // Can't target what we don't know
            Self::Impossible => false,  // Fundamentally impossible
        }
    }

    /// Get the resolution strategy
    pub fn resolution_strategy(&self) -> ResolutionStrategy {
        match self {
            Self::None => ResolutionStrategy::None,
            Self::Known => ResolutionStrategy::Fetch,
            Self::KnownUnknown => ResolutionStrategy::Research,
            Self::Unknown => ResolutionStrategy::Explore,
            Self::Impossible => ResolutionStrategy::Reframe,
        }
    }

    /// Get confidence ceiling for this ignorance type
    ///
    /// Even if we answer, this is the max confidence we can claim. For `None`, the
    /// ceiling applies only to the ignorance category; it does not establish completeness.
    pub fn confidence_ceiling(&self) -> f32 {
        match self {
            // This is a ceiling for the ignorance category, not a proof of completeness.
        Self::None => 1.0,
            Self::Known => 0.95,
            Self::KnownUnknown => 0.85,
            Self::Unknown => 0.50,
            Self::Impossible => 0.10,
        }
    }

    /// Get Greek symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::None => "κ",
            Self::Known => "κ-t",
            Self::KnownUnknown => "ι₁",
            Self::Unknown => "ι₂",
            Self::Impossible => "ι₃",
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No ignorance detected under the active frame",
            Self::Known => "Information exists but is not currently accessible",
            Self::KnownUnknown => "We know what we don't know",
            Self::Unknown => "We don't know what we don't know",
            Self::Impossible => "Fundamentally unknowable",
        }
    }
}

/// Strategy for resolving ignorance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// No resolution needed
    None,
    /// Fetch from existing source
    Fetch,
    /// Active research required
    Research,
    /// Broad exploration needed
    Explore,
    /// Question needs reframing
    Reframe,
}

impl ResolutionStrategy {
    /// Get the expected effort level (0.0 - 1.0)
    pub fn effort_level(&self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Fetch => 0.1,
            Self::Research => 0.5,
            Self::Explore => 0.8,
            Self::Reframe => 0.3,
        }
    }
}

/// Domain of knowledge
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// Mathematics - formal proofs possible
    Mathematics,
    /// Physics - empirical verification possible
    Physics,
    /// History - documentary evidence possible
    History,
    /// Subjective - personal experience
    Subjective,
    /// General knowledge
    General,
    /// Undefined/malformed domain
    Undefined,
}

impl Domain {
    /// Is this domain amenable to formal proof?
    pub fn is_formal(&self) -> bool {
        matches!(self, Domain::Mathematics)
    }

    /// Is this domain empirically verifiable?
    pub fn is_empirical(&self) -> bool {
        matches!(self, Domain::Physics | Domain::History)
    }

    /// Is this domain inherently subjective?
    pub fn is_subjective(&self) -> bool {
        matches!(self, Domain::Subjective)
    }
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Mathematics => write!(f, "mathematics"),
            Domain::Physics => write!(f, "physics"),
            Domain::History => write!(f, "history"),
            Domain::Subjective => write!(f, "subjective"),
            Domain::General => write!(f, "general"),
            Domain::Undefined => write!(f, "undefined"),
        }
    }
}

/// Ignorance record for tracking
#[derive(Debug, Clone)]
pub struct IgnoranceRecord {
    /// Unique identifier
    pub id: String,

    /// The detection result
    pub detection: super::IgnoranceDetection,

    /// Current status
    pub status: IgnoranceStatus,

    /// Resolution (if resolved)
    pub resolution: Option<IgnoranceResolution>,

    /// Append-only frame/correction lineage. Historical entries are never rewritten.
    pub frame_revisions: Vec<EpistemicFrameRevision>,

    /// When created
    pub created_at: SystemTime,

    /// When last updated
    pub updated_at: SystemTime,
}

impl IgnoranceRecord {
    /// Append a frame revision without mutating the historical detection.
    pub fn append_frame_revision(&mut self, revision: EpistemicFrameRevision) {
        self.frame_revisions.push(revision);
        self.updated_at = SystemTime::now();
    }

    /// Append a frame revision only when it continues the current lineage.
    pub fn try_append_frame_revision(
        &mut self,
        revision: EpistemicFrameRevision,
    ) -> Result<(), FrameLineageError> {
        let expected = self
            .latest_frame_revision()
            .map(|entry| entry.revised_frame.as_str().to_owned())
            .unwrap_or_else(|| self.detection.frame.identity());

        if !revision.follows_frame(&expected) {
            return Err(FrameLineageError::Discontinuous);
        }
        if !revision.changes_frame() {
            return Err(FrameLineageError::NoOp);
        }

        self.append_frame_revision(revision);
        Ok(())
    }

    /// Return the most recent frame revision, if any.
    pub fn latest_frame_revision(&self) -> Option<&EpistemicFrameRevision> {
        self.frame_revisions.last()
    }

    /// Verify that the append-only lineage forms one continuous frame chain.
    ///
    /// The first revision must begin at the detection frame, and every later revision
    /// must begin at the prior revision's revised frame. This makes silent provenance
    /// jumps detectable without rewriting historical entries.
    pub fn frame_lineage_is_contiguous(&self) -> bool {
        let mut expected = self.detection.frame.identity();
        for revision in &self.frame_revisions {
            if !revision.follows_frame(&expected) || !revision.changes_frame() {
                return false;
            }
            expected = revision.revised_frame.clone();
        }
        true
    }
}

/// Status of an ignorance record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgnoranceStatus {
    /// Active - still ignorant
    Active,
    /// Resolution has been requested
    ResolutionRequested,
    /// Successfully resolved
    Resolved,
    /// Expired without resolution
    Expired,
}

/// Resolution of an ignorance
#[derive(Debug, Clone)]
pub struct IgnoranceResolution {
    /// How was it resolved?
    pub method: ResolutionMethod,

    /// The answer (if applicable)
    pub answer: Option<String>,

    /// Confidence in the resolution
    pub confidence: f32,

    /// Source of resolution
    pub source: String,

    /// When resolved
    pub resolved_at: SystemTime,
}

/// Method of resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMethod {
    /// Found in local knowledge
    LocalKnowledge,
    /// Retrieved from network
    NetworkRetrieval,
    /// Derived through reasoning
    Derivation,
    /// Received from Dark Spot DHT
    DarkSpotMatch,
    /// User provided answer
    UserProvided,
    /// Question was reframed
    Reframed,
}

// =============================================================================
// Epistemic frame provenance
// =============================================================================

/// The representational frame under which a GIS conclusion was produced.
///
/// A frame qualifies confidence: it records the evidence/model/schema boundary
/// rather than pretending that a high proposition confidence proves completeness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicFrame {
    /// Stable semantic identifier for the frame definition.
    pub id: String,
    /// Monotonic schema/frame revision.
    pub version: u32,
    /// Boundary of evidence considered by the frame.
    pub evidence_boundary: String,
    /// Ontology/schema identifier used to interpret observations.
    pub ontology_id: String,
    /// Causal/model identifier used for inference.
    pub causal_model_id: String,
    /// Variables deliberately excluded from the frame.
    pub excluded_variables: Vec<String>,
    /// Known blind spots acknowledged by the frame.
    pub known_blind_spots: Vec<String>,
}

impl EpistemicFrame {
    /// Stable identity used for provenance and future correction events.
    pub fn identity(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }

    /// Compare this frame with another frame without collapsing either into a scalar score.
    ///
    /// A divergent ontology or causal model is itself epistemically relevant: conclusions
    /// produced under the two frames must not be treated as interchangeable merely because
    /// their proposition-level confidence happens to be similar.
    pub fn divergence_from(&self, other: &Self) -> EpistemicFrameDivergence {
        EpistemicFrameDivergence {
            version_changed: self.version != other.version,
            evidence_boundary_changed: self.evidence_boundary != other.evidence_boundary,
            ontology_changed: self.ontology_id != other.ontology_id,
            causal_model_changed: self.causal_model_id != other.causal_model_id,
            excluded_variables_changed: self.excluded_variables != other.excluded_variables,
            blind_spots_changed: self.known_blind_spots != other.known_blind_spots,
        }
    }
}

/// Structured difference between two epistemic frames.
///
/// This intentionally avoids a single "frame uncertainty" number. Different kinds of
/// frame divergence have different meanings and should remain inspectable for provenance,
/// correction, and later model comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicFrameDivergence {
    /// Frame schema revision changed; provenance must therefore remain distinct.
    pub version_changed: bool,
    pub evidence_boundary_changed: bool,
    pub ontology_changed: bool,
    pub causal_model_changed: bool,
    pub excluded_variables_changed: bool,
    pub blind_spots_changed: bool,
}

impl EpistemicFrameDivergence {
    /// Whether any epistemically material frame component differs.
    pub fn is_divergent(&self) -> bool {
        self.version_changed
            || self.evidence_boundary_changed
            || self.ontology_changed
            || self.causal_model_changed
            || self.excluded_variables_changed
            || self.blind_spots_changed
    }
}


/// Append-only record describing a revision of an epistemic frame.
///
/// A revision never mutates or erases conclusions produced under the prior frame.
/// It records why the frame changed and which previously derived conclusions require
/// re-evaluation or scope qualification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicFrameRevision {
    /// Prior frame identity (id@version).
    pub prior_frame: String,
    /// Revised frame identity (id@version).
    pub revised_frame: String,
    /// Evidence or observation that triggered the revision.
    pub trigger: String,
    /// Newly represented entity, variable, or relationship, if any.
    pub newly_represented: Option<String>,
    /// Human/model-readable scope change.
    pub scope_change: String,
    /// Claim or conclusion identifiers affected by the revision.
    pub affected_conclusions: Vec<String>,
    /// Typed impact mask derived from the prior/revised frame pair.
    pub impact: EpistemicFrameImpact,
}

impl EpistemicFrameRevision {
    /// Construct an append-only frame revision event.
    pub fn new(
        prior_frame: &EpistemicFrame,
        revised_frame: &EpistemicFrame,
        trigger: impl Into<String>,
        newly_represented: Option<String>,
        scope_change: impl Into<String>,
        affected_conclusions: Vec<String>,
    ) -> Self {
        Self {
            prior_frame: prior_frame.identity(),
            revised_frame: revised_frame.identity(),
            trigger: trigger.into(),
            newly_represented,
            scope_change: scope_change.into(),
            affected_conclusions,
            impact: EpistemicFrameImpact::from_divergence(
                &prior_frame.divergence_from(revised_frame),
            ),
        }
    }

    /// Whether this revision continues directly from the supplied frame identity.
    pub fn follows_frame(&self, frame_identity: &str) -> bool {
        self.prior_frame == frame_identity
    }

    /// Whether this revision is structurally meaningful rather than a no-op.
    pub fn changes_frame(&self) -> bool {
        self.prior_frame != self.revised_frame
    }
}

/// Lifecycle state of a conclusion whose provenance may be affected by frame revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConclusionStatus {
    /// Conclusion remains usable under its originating frame.
    Active,
    /// Conclusion remains historical but requires qualification under a changed frame.
    Qualified,
    /// Conclusion must be evaluated again before being used as current knowledge.
    Reopened,
    /// Conclusion has been explicitly replaced by a later conclusion.
    Superseded,
}

/// Why a downstream conclusion depends on an upstream conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConclusionDependencyKind {
    EvidenceSupport,
    CausalDependency,
    DefinitionDependency,
    OntologyDependency,
    InferenceDependency,
    AssumptionDependency,
}

/// Which dependency kinds are potentially affected by a frame revision.
///
/// This is deliberately a capability mask rather than a scalar severity score.
/// A revision asks which relationships require re-evaluation; it does not assert
/// that every downstream conclusion is false.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicFrameImpact {
    pub evidence_boundary: bool,
    pub ontology: bool,
    pub causal_model: bool,
    pub exclusions: bool,
    pub blind_spots: bool,
}

impl EpistemicFrameImpact {
    pub const fn broad() -> Self {
        Self {
            evidence_boundary: true,
            ontology: true,
            causal_model: true,
            exclusions: true,
            blind_spots: true,
        }
    }

    /// Derive dependency impact from a structured frame divergence.
    pub fn from_divergence(divergence: &EpistemicFrameDivergence) -> Self {
        let material = divergence.evidence_boundary_changed
            || divergence.ontology_changed
            || divergence.causal_model_changed
            || divergence.excluded_variables_changed
            || divergence.blind_spots_changed;

        if !material {
            // A version-only change is still a provenance boundary. Without a
            // semantic diff, prefer re-evaluation over silently trusting old edges.
            return Self::broad();
        }

        Self {
            evidence_boundary: divergence.evidence_boundary_changed,
            ontology: divergence.ontology_changed,
            causal_model: divergence.causal_model_changed,
            exclusions: divergence.excluded_variables_changed,
            blind_spots: divergence.blind_spots_changed,
        }
    }

    pub fn affects(&self, kind: ConclusionDependencyKind) -> bool {
        match kind {
            ConclusionDependencyKind::EvidenceSupport => self.evidence_boundary,
            ConclusionDependencyKind::CausalDependency => self.causal_model,
            ConclusionDependencyKind::DefinitionDependency => self.ontology,
            ConclusionDependencyKind::OntologyDependency => self.ontology,
            ConclusionDependencyKind::InferenceDependency => {
                self.evidence_boundary || self.ontology || self.causal_model
            }
            ConclusionDependencyKind::AssumptionDependency => {
                self.evidence_boundary
                    || self.ontology
                    || self.causal_model
                    || self.exclusions
                    || self.blind_spots
            }
        }
    }
}

/// Typed dependency between epistemic conclusions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConclusionDependency {
    pub upstream: String,
    pub downstream: String,
    pub kind: ConclusionDependencyKind,
}

impl ConclusionDependency {
    pub fn new(
        upstream: impl Into<String>,
        downstream: impl Into<String>,
        kind: ConclusionDependencyKind,
    ) -> Self {
        Self { upstream: upstream.into(), downstream: downstream.into(), kind }
    }
}

/// A frame-qualified epistemic conclusion.
///
/// This is intentionally narrower than a general knowledge-graph node: it records the
/// provenance needed to reopen reasoning when its frame or an upstream conclusion changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicConclusion {
    /// Stable conclusion identifier.
    pub id: String,
    /// Human/model-readable proposition represented by this conclusion.
    pub proposition: String,
    /// Frame under which the conclusion was produced.
    pub originating_frame: String,
    /// Evidence identifiers directly supporting the conclusion.
    pub evidence: Vec<String>,
    /// Upstream conclusion identifiers required by this conclusion.
    pub dependencies: Vec<String>,
    /// Current lifecycle state.
    pub status: ConclusionStatus,
}

impl EpistemicConclusion {
    pub fn new(
        id: impl Into<String>,
        proposition: impl Into<String>,
        originating_frame: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            proposition: proposition.into(),
            originating_frame: originating_frame.into(),
            evidence: Vec::new(),
            dependencies: Vec::new(),
            status: ConclusionStatus::Active,
        }
    }

    /// Mark a historical conclusion as requiring re-evaluation under a changed frame.
    pub fn reopen(&mut self) {
        // Superseded conclusions remain historical replacements; reopening them would
        // blur the distinction between correction of a live belief and resurrection of
        // an explicitly replaced one.
        if self.status != ConclusionStatus::Superseded {
            self.status = ConclusionStatus::Reopened;
        }
    }

    /// Preserve the conclusion while explicitly qualifying it against its originating frame.
    pub fn qualify(&mut self) {
        if self.status != ConclusionStatus::Reopened {
            self.status = ConclusionStatus::Qualified;
        }
    }
}

/// Minimal dependency graph for deterministic downstream impact analysis.
///
/// The graph deliberately does not infer semantic validity. It only answers the narrower
/// provenance question: which conclusions transitively depend on conclusions whose frame
/// provenance has changed?
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConclusionDependencyGraph {
    pub conclusions: Vec<EpistemicConclusion>,
    /// Typed dependency edges; legacy conclusion-local dependency IDs remain supported.
    pub dependency_edges: Vec<ConclusionDependency>,
}

impl ConclusionDependencyGraph {
    pub fn add(&mut self, conclusion: EpistemicConclusion) {
        self.conclusions.push(conclusion);
    }

    pub fn add_dependency(&mut self, dependency: ConclusionDependency) {
        self.dependency_edges.push(dependency);
    }

    /// Return conclusion IDs transitively downstream of the supplied roots.
    pub fn downstream_of(&self, roots: &[String]) -> Vec<String> {
        let mut affected = Vec::new();
        let mut frontier = roots.to_vec();

        while let Some(root) = frontier.pop() {
            for dependency in &self.dependency_edges {
                if dependency.upstream == root
                    && !affected.iter().any(|id| id == &dependency.downstream)
                    && !roots.iter().any(|id| id == &dependency.downstream)
                {
                    affected.push(dependency.downstream.clone());
                    frontier.push(dependency.downstream.clone());
                }
            }

            for conclusion in &self.conclusions {
                if conclusion.dependencies.iter().any(|dep| dep == &root)
                    && !affected.iter().any(|id| id == &conclusion.id)
                    && !roots.iter().any(|id| id == &conclusion.id)
                {
                    affected.push(conclusion.id.clone());
                    frontier.push(conclusion.id.clone());
                }
            }
        }

        affected
    }

    pub fn dependency_reasons(
        &self,
        upstream: &str,
        downstream: &str,
    ) -> Vec<ConclusionDependencyKind> {
        self.dependency_edges
            .iter()
            .filter(|edge| edge.upstream == upstream && edge.downstream == downstream)
            .map(|edge| edge.kind)
            .collect()
    }

    /// Reopen roots and all transitively dependent conclusions.
    pub fn reopen_from(&mut self, roots: &[String]) -> Vec<String> {
        // Only return conclusions that actually exist in the graph. A provenance event
        // may reference a stale/deleted ID; silently reporting it as reopened would turn
        // missing provenance into false evidence of correction.
        let known: std::collections::HashSet<&str> =
            self.conclusions.iter().map(|c| c.id.as_str()).collect();

        let downstream = self.downstream_of(roots);
        let mut candidates = roots.to_vec();
        candidates.extend(downstream);

        let mut reopened = Vec::new();
        for id in candidates {
            if known.contains(id.as_str()) && !reopened.iter().any(|seen| seen == &id) {
                reopened.push(id);
            }
        }

        for conclusion in &mut self.conclusions {
            if reopened.iter().any(|id| id == &conclusion.id) {
                conclusion.reopen();
            }
        }

        reopened
    }

    /// Apply a frame revision using dependency semantics rather than graph proximity.
    /// Typed edges propagate only when their dependency kind is affected. Legacy untyped
    /// dependency IDs remain conservative because their semantic basis is unavailable.
    pub fn reopen_from_frame_revision(
        &mut self,
        revision: &EpistemicFrameRevision,
    ) -> Vec<String> {
        let mut affected = Vec::new();
        let mut frontier = Vec::new();
        let known: std::collections::HashSet<&str> =
            self.conclusions.iter().map(|c| c.id.as_str()).collect();

        for root in &revision.affected_conclusions {
            if known.contains(root.as_str()) && !affected.contains(root) {
                affected.push(root.clone());
                frontier.push(root.clone());
            }
        }

        while let Some(upstream) = frontier.pop() {
            for edge in &self.dependency_edges {
                if edge.upstream == upstream
                    && revision.impact.affects(edge.kind)
                    && known.contains(edge.downstream.as_str())
                    && !affected.contains(&edge.downstream)
                {
                    affected.push(edge.downstream.clone());
                    frontier.push(edge.downstream.clone());
                }
            }

            for conclusion in &self.conclusions {
                let depends = conclusion.dependencies.iter().any(|id| id == &upstream);
                let has_typed_edge = self.dependency_edges.iter().any(|edge| {
                    edge.upstream == upstream && edge.downstream == conclusion.id
                });
                if depends && !has_typed_edge && !affected.contains(&conclusion.id) {
                    affected.push(conclusion.id.clone());
                    frontier.push(conclusion.id.clone());
                }
            }
        }

        for conclusion in &mut self.conclusions {
            if affected.contains(&conclusion.id) {
                conclusion.reopen();
            }
        }
        affected
    }
}

/// Failure returned when a frame revision would corrupt append-only lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameLineageError {
    /// The revision starts from a different frame than the record's current frame.
    Discontinuous,
    /// The revision does not actually change the frame.
    NoOp,
}

impl std::fmt::Display for FrameLineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Discontinuous => write!(f, "frame revision is discontinuous"),
            Self::NoOp => write!(f, "frame revision is a no-op"),
        }
    }
}

impl std::error::Error for FrameLineageError {}

impl Default for EpistemicFrame {
    fn default() -> Self {
        Self {
            id: "gis-default".to_string(),
            version: 1,
            evidence_boundary: "local-query-context".to_string(),
            ontology_id: "general-v1".to_string(),
            causal_model_id: "unspecified".to_string(),
            excluded_variables: Vec::new(),
            known_blind_spots: vec!["unrepresented variables and categories".to_string()],
        }
    }
}

// =============================================================================
// GIS v4.0: Eight Harmonies Integration
// =============================================================================

// Canonical Harmony type from shared types crate
pub use symthaea_types::Harmony;

/// Harmonic Ignorance: Ignorance weighted by affected harmonies
///
/// This is the core GIS v4.0 extension. Ignorance is not uniform -
/// a gap in Care-Knowing differs fundamentally from a gap in Truth-Knowing.
#[derive(Debug, Clone)]
pub struct HarmonicIgnorance {
    /// The base ignorance type
    pub base_ignorance: IgnoranceType,

    /// Which harmonies are affected and by how much (0.0 - 1.0)
    pub affected_harmonies: Vec<(Harmony, f32)>,

    /// When this harmonic ignorance was detected
    pub detected_at: SystemTime,
}

impl HarmonicIgnorance {
    /// Create a new harmonic ignorance with base type
    pub fn new(base_ignorance: IgnoranceType) -> Self {
        Self {
            base_ignorance,
            affected_harmonies: Vec::new(),
            detected_at: SystemTime::now(),
        }
    }

    /// Add an affected harmony with its impact level
    pub fn add_affected_harmony(&mut self, harmony: Harmony, impact: f32) {
        let impact = impact.clamp(0.0, 1.0);
        // Update if already present, otherwise add
        if let Some(entry) = self
            .affected_harmonies
            .iter_mut()
            .find(|(h, _)| *h == harmony)
        {
            entry.1 = impact;
        } else {
            self.affected_harmonies.push((harmony, impact));
        }
    }

    /// Calculate total harmonic impact (weighted sum)
    pub fn total_harmonic_impact(&self) -> f32 {
        self.affected_harmonies
            .iter()
            .map(|(harmony, impact)| harmony.base_weight() * impact)
            .sum()
    }

    /// Get the primary (most affected) harmony gap
    pub fn primary_harmony_gap(&self) -> Option<Harmony> {
        self.affected_harmonies
            .iter()
            .max_by(|a, b| {
                let weight_a = a.0.base_weight() * a.1;
                let weight_b = b.0.base_weight() * b.1;
                weight_a
                    .partial_cmp(&weight_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(h, _)| *h)
    }

    /// Get harmonies above a threshold impact
    pub fn significant_gaps(&self, threshold: f32) -> Vec<Harmony> {
        self.affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact >= threshold)
            .map(|(h, _)| *h)
            .collect()
    }

    /// Generate resolution paths based on affected harmonies
    pub fn resolution_paths(&self) -> Vec<HarmonicResolution> {
        self.affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact > 0.3) // Only significant gaps
            .map(|(harmony, impact)| HarmonicResolution {
                harmony: *harmony,
                impact: *impact,
                strategy: harmony_resolution_strategy(*harmony, self.base_ignorance),
                estimated_effort: harmony_resolution_effort(*harmony, *impact),
            })
            .collect()
    }

    /// Generate the H-dimension code for E/N/M/H notation
    pub fn h_code(&self) -> String {
        if self.affected_harmonies.is_empty() {
            return String::from("H{}");
        }

        let parts: Vec<String> = self
            .affected_harmonies
            .iter()
            .filter(|(_, impact)| *impact > 0.1) // Only show significant impacts
            .map(|(h, impact)| format!("{}:{:.1}", h.code(), impact))
            .collect();

        format!("H{{{}}}", parts.join(","))
    }

    /// Calculate confidence ceiling based on harmonic impacts
    ///
    /// High impact on PSF or IW lowers confidence ceiling.
    pub fn confidence_ceiling(&self) -> f32 {
        let base_ceiling = self.base_ignorance.confidence_ceiling();

        // Critical harmonies reduce ceiling more
        let critical_penalty: f32 = self
            .affected_harmonies
            .iter()
            .filter(|(h, _)| matches!(h, Harmony::PanSentientFlourishing | Harmony::IntegralWisdom))
            .map(|(_, impact)| impact * 0.2)
            .sum();

        (base_ceiling - critical_penalty).max(0.1)
    }
}

/// Resolution path for a specific harmony gap
#[derive(Debug, Clone)]
pub struct HarmonicResolution {
    /// The harmony to address
    pub harmony: Harmony,
    /// Current impact level
    pub impact: f32,
    /// Recommended resolution strategy
    pub strategy: HarmonicResolutionStrategy,
    /// Estimated effort (0.0 - 1.0)
    pub estimated_effort: f32,
}

/// Harmony-specific resolution strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicResolutionStrategy {
    /// Seek integration, find connecting patterns
    SeekIntegration,
    /// Consult affected parties, gather perspectives
    ConsultAffected,
    /// Verify facts, seek evidence
    VerifyEvidence,
    /// Explore possibilities, brainstorm
    ExplorePossibilities,
    /// Map relationships, trace connections
    MapRelationships,
    /// Consider exchange dynamics, reciprocity
    AnalyzeReciprocity,
    /// Study development patterns, emergence
    StudyEmergence,
    /// General research
    Research,
}

/// Determine resolution strategy based on harmony and ignorance type
fn harmony_resolution_strategy(
    harmony: Harmony,
    ignorance: IgnoranceType,
) -> HarmonicResolutionStrategy {
    match (harmony, ignorance) {
        (Harmony::ResonantCoherence, _) => HarmonicResolutionStrategy::SeekIntegration,
        (Harmony::PanSentientFlourishing, _) => HarmonicResolutionStrategy::ConsultAffected,
        (Harmony::IntegralWisdom, _) => HarmonicResolutionStrategy::VerifyEvidence,
        (Harmony::InfinitePlay, _) => HarmonicResolutionStrategy::ExplorePossibilities,
        (Harmony::UniversalInterconnectedness, _) => HarmonicResolutionStrategy::MapRelationships,
        (Harmony::SacredReciprocity, _) => HarmonicResolutionStrategy::AnalyzeReciprocity,
        (Harmony::EvolutionaryProgression, _) => HarmonicResolutionStrategy::StudyEmergence,
        (Harmony::SacredStillness, _) => HarmonicResolutionStrategy::SeekIntegration,
    }
}

/// Estimate resolution effort based on harmony and impact
fn harmony_resolution_effort(harmony: Harmony, impact: f32) -> f32 {
    let base_effort = match harmony {
        // Social harmonies require more effort (stakeholder engagement)
        Harmony::PanSentientFlourishing => 0.7,
        Harmony::UniversalInterconnectedness => 0.6,
        // Verification takes moderate effort
        Harmony::IntegralWisdom => 0.5,
        // Integration can be quick if data exists
        Harmony::ResonantCoherence => 0.4,
        // Creative exploration is variable
        Harmony::InfinitePlay => 0.3,
        // Pattern recognition can be efficient
        Harmony::SacredReciprocity => 0.4,
        Harmony::EvolutionaryProgression => 0.5,
        Harmony::SacredStillness => 0.3,
    };

    (base_effort * impact).clamp(0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_lineage_contiguity() {
        let mut record = IgnoranceRecord {
            id: "lineage".to_string(),
            detection: super::IgnoranceDetection {
                query: "q".to_string(),
                ignorance_type: IgnoranceType::KnownUnknown,
                uncertainty: Uncertainty3D::new(0.2, 0.2, 0.2),
                domain: Domain::General,
                eig: 0.5,
                detected_at: SystemTime::now(),
                frame: EpistemicFrame::default(),
            },
            status: IgnoranceStatus::Active,
            resolution: None,
            frame_revisions: Vec::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };
        assert!(record.frame_lineage_is_contiguous());

        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame { version: 2, ..prior.clone() };
        record.append_frame_revision(EpistemicFrameRevision::new(
            &prior, &revised, "revision-1", None, "version bump", vec![],
        ));
        assert!(record.frame_lineage_is_contiguous());

        let next = EpistemicFrame { version: 3, ..revised.clone() };
        record.append_frame_revision(EpistemicFrameRevision::new(
            &revised, &next, "revision-2", None, "version bump", vec![],
        ));
        assert!(record.frame_lineage_is_contiguous());
    }

    #[test]
    fn test_ignorance_resolvability() {
        assert!(IgnoranceType::None.is_resolvable());
        assert!(IgnoranceType::Known.is_resolvable());
        assert!(IgnoranceType::KnownUnknown.is_resolvable());
        assert!(!IgnoranceType::Unknown.is_resolvable());
        assert!(!IgnoranceType::Impossible.is_resolvable());
    }

    #[test]
    fn test_confidence_ceilings() {
        assert!(
            IgnoranceType::None.confidence_ceiling()
                > IgnoranceType::KnownUnknown.confidence_ceiling()
        );
        assert!(
            IgnoranceType::KnownUnknown.confidence_ceiling()
                > IgnoranceType::Unknown.confidence_ceiling()
        );
        assert!(
            IgnoranceType::Unknown.confidence_ceiling()
                > IgnoranceType::Impossible.confidence_ceiling()
        );
    }

    #[test]
    fn test_conclusion_dependency_reopening_is_transitive() {
        let mut graph = ConclusionDependencyGraph::default();

        let root = EpistemicConclusion::new("c1", "A", "gis-default@1");
        let mut dependent = EpistemicConclusion::new("c2", "B", "gis-default@1");
        dependent.dependencies.push("c1".to_string());
        let mut downstream = EpistemicConclusion::new("c3", "C", "gis-default@1");
        downstream.dependencies.push("c2".to_string());

        graph.add(root);
        graph.add(dependent);
        graph.add(downstream);
        graph.add_dependency(ConclusionDependency::new(
            "c1",
            "c2",
            ConclusionDependencyKind::OntologyDependency,
        ));

        let reopened = graph.reopen_from(&["c1".to_string()]);
        assert_eq!(reopened, vec!["c1".to_string(), "c2".to_string(), "c3".to_string()]);
        assert!(graph.conclusions.iter().all(|c| c.status == ConclusionStatus::Reopened));
        assert_eq!(
            graph.dependency_reasons("c1", "c2"),
            vec![ConclusionDependencyKind::OntologyDependency]
        );
    }

    #[test]
    fn test_frame_revision_propagates_to_typed_conclusions() {
        let prior = EpistemicFrame::default();
        let revised = EpistemicFrame { version: 2, ..prior.clone() };
        let revision = EpistemicFrameRevision::new(
            &prior,
            &revised,
            "new variable represented",
            Some("institutional-role".to_string()),
            "expanded ontology",
            vec!["c1".to_string()],
        );

        let mut graph = ConclusionDependencyGraph::default();
        graph.add(EpistemicConclusion::new("c1", "root", prior.identity()));
        let mut c2 = EpistemicConclusion::new("c2", "dependent", prior.identity());
        c2.dependencies.push("c1".to_string());
        graph.add(c2);

        let reopened = graph.reopen_from_frame_revision(&revision);
        assert_eq!(reopened, vec!["c1".to_string(), "c2".to_string()]);
        assert!(graph.conclusions.iter().all(|c| c.status == ConclusionStatus::Reopened));
    }

    #[test]
    fn test_checked_frame_revision_append_rejects_invalid_history() {
        let mut record = IgnoranceRecord {
            id: "checked-lineage".to_string(),
            detection: super::IgnoranceDetection {
                query: "q".to_string(),
                ignorance_type: IgnoranceType::KnownUnknown,
                uncertainty: Uncertainty3D::new(0.2, 0.2, 0.2),
                domain: Domain::General,
                eig: 0.5,
                detected_at: SystemTime::now(),
                frame: EpistemicFrame::default(),
            },
            status: IgnoranceStatus::Active,
            resolution: None,
            frame_revisions: Vec::new(),
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        };
        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame { version: 2, ..prior.clone() };

        let valid = EpistemicFrameRevision::new(
            &prior, &revised, "valid", None, "version bump", vec![],
        );
        assert!(record.try_append_frame_revision(valid).is_ok());

        let no_op = EpistemicFrameRevision::new(
            &revised, &revised, "noop", None, "unchanged", vec![],
        );
        assert_eq!(
            record.try_append_frame_revision(no_op),
            Err(FrameLineageError::NoOp)
        );

        let discontinuous = EpistemicFrameRevision::new(
            &prior,
            &EpistemicFrame { version: 3, ..prior.clone() },
            "stale writer",
            None,
            "skipped current frame",
            vec![],
        );
        assert_eq!(
            record.try_append_frame_revision(discontinuous),
            Err(FrameLineageError::Discontinuous)
        );
        assert_eq!(record.frame_revisions.len(), 1);
        assert!(record.frame_lineage_is_contiguous());
    }

    #[test]
    fn test_frame_revision_continuity_and_non_noop() {
        let prior = EpistemicFrame::default();
        let revised = EpistemicFrame {
            version: 2,
            ontology_id: "collective-agents-v2".to_string(),
            ..prior.clone()
        };
        let revision = EpistemicFrameRevision::new(
            &prior,
            &revised,
            "new relationship observed",
            Some("institutional-role".to_string()),
            "expanded ontology",
            vec!["conclusion-1".to_string()],
        );

        assert!(revision.follows_frame(&prior.identity()));
        assert!(revision.changes_frame());
        assert!(!revision.follows_frame("other@1"));
    }

    #[test]
    fn test_frame_impact_is_dependency_sensitive() {
        let prior = EpistemicFrame::default();
        let revised = EpistemicFrame {
            version: 2,
            ontology_id: "collective-agents-v2".to_string(),
            ..prior.clone()
        };
        let revision = EpistemicFrameRevision::new(
            &prior,
            &revised,
            "ontology expanded",
            Some("institutional-role".to_string()),
            "ontology change",
            vec!["c1".to_string()],
        );

        assert!(revision.impact.ontology);
        assert!(!revision.impact.causal_model);
        assert!(revision.impact.affects(ConclusionDependencyKind::OntologyDependency));
        assert!(!revision.impact.affects(ConclusionDependencyKind::CausalDependency));

        let mut graph = ConclusionDependencyGraph::default();
        graph.add(EpistemicConclusion::new("c1", "root", prior.identity()));
        graph.add(EpistemicConclusion::new("c2", "causal dependent", prior.identity()));
        graph.add(EpistemicConclusion::new("c3", "ontology dependent", prior.identity()));
        graph.add_dependency(ConclusionDependency::new(
            "c1", "c2", ConclusionDependencyKind::CausalDependency,
        ));
        graph.add_dependency(ConclusionDependency::new(
            "c1", "c3", ConclusionDependencyKind::OntologyDependency,
        ));

        let reopened = graph.reopen_from_frame_revision(&revision);
        assert_eq!(reopened, vec!["c1".to_string(), "c3".to_string()]);
        assert_eq!(graph.conclusions[0].status, ConclusionStatus::Reopened);
        assert_eq!(graph.conclusions[1].status, ConclusionStatus::Active);
        assert_eq!(graph.conclusions[2].status, ConclusionStatus::Reopened);
    }

    #[test]
    fn test_reopen_preserves_already_superseded_state() {
        let mut graph = ConclusionDependencyGraph::default();
        let mut c = EpistemicConclusion::new("c1", "A", "gis-default@1");
        c.status = ConclusionStatus::Superseded;
        graph.add(c);

        let reopened = graph.reopen_from(&["c1".to_string()]);
        assert_eq!(reopened, vec!["c1".to_string()]);
        assert_eq!(graph.conclusions[0].status, ConclusionStatus::Superseded);
    }

    #[test]
    fn test_reopen_does_not_report_missing_roots {
        let mut graph = ConclusionDependencyGraph::default();
        graph.add(EpistemicConclusion::new("c1", "A", "gis-default@1"));

        let reopened = graph.reopen_from(&["missing".to_string()]);
        assert!(reopened.is_empty());
        assert_eq!(graph.conclusions[0].status, ConclusionStatus::Active);
    }

    #[test]
    fn test_typed_dependency_cycle_terminates_deterministically() {
        let mut graph = ConclusionDependencyGraph::default();
        graph.add(EpistemicConclusion::new("c1", "A", "gis-default@1"));
        graph.add(EpistemicConclusion::new("c2", "B", "gis-default@1"));
        graph.add_dependency(ConclusionDependency::new(
            "c1", "c2", ConclusionDependencyKind::InferenceDependency,
        ));
        graph.add_dependency(ConclusionDependency::new(
            "c2", "c1", ConclusionDependencyKind::AssumptionDependency,
        ));

        let reopened = graph.reopen_from(&["c1".to_string()]);
        assert_eq!(reopened, vec!["c1".to_string(), "c2".to_string()]);
    }

    #[test]
    fn test_domain_classification() {
        assert!(Domain::Mathematics.is_formal());
        assert!(!Domain::Mathematics.is_empirical());
        assert!(Domain::Physics.is_empirical());
        assert!(Domain::Subjective.is_subjective());
    }

    #[test]
    fn test_symbols() {
        assert_eq!(IgnoranceType::None.symbol(), "κ");
        assert_eq!(IgnoranceType::KnownUnknown.symbol(), "ι₁");
        assert_eq!(IgnoranceType::Unknown.symbol(), "ι₂");
        assert_eq!(IgnoranceType::Impossible.symbol(), "ι₃");
    }

    // === GIS v4.0 Harmony Tests ===

    #[test]
    fn test_harmony_weights_sum_to_one() {
        let total: f32 = Harmony::all().iter().map(|h| h.base_weight()).sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Harmony weights should sum to ~1.0, got {}",
            total
        );
    }

    #[test]
    fn test_harmony_codes() {
        assert_eq!(Harmony::ResonantCoherence.code(), "RC");
        assert_eq!(Harmony::PanSentientFlourishing.code(), "PSF");
        assert_eq!(Harmony::IntegralWisdom.code(), "IW");
    }

    #[test]
    fn test_harmonic_ignorance_creation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
        hi.add_affected_harmony(Harmony::IntegralWisdom, 0.6);

        assert_eq!(hi.affected_harmonies.len(), 2);
        assert_eq!(
            hi.primary_harmony_gap(),
            Some(Harmony::PanSentientFlourishing)
        );
    }

    #[test]
    fn test_harmonic_impact_calculation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 1.0); // weight 0.17

        let impact = hi.total_harmonic_impact();
        assert!(
            (impact - 0.17).abs() < 0.01,
            "Impact should be ~0.17, got {}",
            impact
        );
    }

    #[test]
    fn test_h_code_generation() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::ResonantCoherence, 0.8);
        hi.add_affected_harmony(Harmony::InfinitePlay, 0.3);

        let code = hi.h_code();
        assert!(
            code.contains("RC:0.8"),
            "Should contain RC:0.8, got {}",
            code
        );
        assert!(
            code.contains("IP:0.3"),
            "Should contain IP:0.3, got {}",
            code
        );
    }

    #[test]
    fn test_resolution_paths() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
        hi.add_affected_harmony(Harmony::IntegralWisdom, 0.5);
        hi.add_affected_harmony(Harmony::InfinitePlay, 0.1); // Too low, should be excluded

        let paths = hi.resolution_paths();
        assert_eq!(paths.len(), 2); // Only PSF and IW
        assert!(
            paths
                .iter()
                .any(|p| p.harmony == Harmony::PanSentientFlourishing)
        );
        assert!(paths.iter().any(|p| p.harmony == Harmony::IntegralWisdom));
    }

    #[test]
    fn test_harmonic_confidence_ceiling() {
        let mut hi = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        let base_ceiling = hi.base_ignorance.confidence_ceiling();

        // Add critical harmony impact
        hi.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);

        let adjusted_ceiling = hi.confidence_ceiling();
        assert!(
            adjusted_ceiling < base_ceiling,
            "Critical harmony should reduce ceiling"
        );
    }
}
