// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Bridge Integrity Zome
//!
//! Entry types for cross-hApp knowledge queries and claim verification.
//! Enhanced with GIS v4.0 (Graceful Ignorance System) integration for
//! epistemic classification, multi-perspective analysis, and harmonic alignment.
//!
//! ## GIS v4.0 Integration
//!
//! This module bridges Mycelix epistemic claims to the Symthaea Graceful Ignorance
//! System, enabling:
//! - 5-type ignorance classification (κ, ι₁, ι₂, ι₃, ι∞)
//! - 3D uncertainty quantification (epistemic, aleatoric, structural)
//! - Multi-perspective Rashomon analysis via Eight Harmonies
//! - Dark Spot DHT for collective ignorance resolution
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// =============================================================================
// EXISTING KNOWLEDGE BRIDGE TYPES
// =============================================================================

/// Knowledge query from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KnowledgeQuery {
    pub id: String,
    pub query_type: KnowledgeQueryType,
    pub source_happ: String,
    pub parameters: String,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum KnowledgeQueryType {
    VerifyClaim,
    ClaimsBySubject,
    EpistemicScore,
    GraphTraversal,
    FactCheck,
    // GIS v4.0 query types
    GisClassify,
    RashomonAnalyze,
    DarkSpotQuery,
    HarmonicAlignment,
}

/// Epistemic classification result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EpistemicResult {
    pub id: String,
    pub claim_id: String,
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
    pub credibility: f64,
    pub source_count: u32,
    pub verified_at: Timestamp,
}

/// Cross-hApp claim reference
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClaimReference {
    pub claim_id: String,
    pub source_happ: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub epistemic_e: f64,
    pub epistemic_n: f64,
    pub epistemic_m: f64,
    pub created_at: Timestamp,
}

/// Bridge event for knowledge updates
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KnowledgeBridgeEvent {
    pub id: String,
    pub event_type: KnowledgeEventType,
    pub claim_id: Option<String>,
    pub subject: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum KnowledgeEventType {
    ClaimCreated,
    ClaimUpdated,
    ClaimDisputed,
    ClaimVerified,
    GraphUpdated,
    InferenceCompleted,
    // GIS v4.0 event types
    GisClassified,
    IgnoranceDetected,
    DarkSpotPublished,
    DarkSpotResolved,
    RashomonAnalyzed,
    HarmonicAligned,
}

// =============================================================================
// GIS v4.0 TYPES - Graceful Ignorance System Integration
// =============================================================================

/// The 5 types of ignorance (κ, ι₁, ι₂, ι₃, ι∞)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgnoranceType {
    /// κ (Known): Information exists and is accessible
    Known,
    /// ι₁ (Known Unknown): Know that we don't know
    KnownUnknown,
    /// ι₂ (Unknown Unknown): Don't know what we don't know
    UnknownUnknown,
    /// ι₃ (Structural Unknown): Unknowable due to structural limits
    StructuralUnknown,
    /// ι∞ (Impossible): Fundamentally unanswerable
    Impossible,
}

impl IgnoranceType {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Known => "κ",
            Self::KnownUnknown => "ι₁",
            Self::UnknownUnknown => "ι₂",
            Self::StructuralUnknown => "ι₃",
            Self::Impossible => "ι∞",
        }
    }

    pub fn resolvability(&self) -> f64 {
        match self {
            Self::Known => 1.0,
            Self::KnownUnknown => 0.7,
            Self::UnknownUnknown => 0.3,
            Self::StructuralUnknown => 0.1,
            Self::Impossible => 0.0,
        }
    }
}

/// 3D Uncertainty quantification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Uncertainty3D {
    /// Epistemic uncertainty: lack of knowledge (reducible)
    pub epistemic: f64,
    /// Aleatoric uncertainty: inherent randomness (irreducible)
    pub aleatoric: f64,
    /// Structural uncertainty: system/model limitations
    pub structural: f64,
}

impl Uncertainty3D {
    pub fn new(epistemic: f64, aleatoric: f64, structural: f64) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            aleatoric: aleatoric.clamp(0.0, 1.0),
            structural: structural.clamp(0.0, 1.0),
        }
    }

    pub fn total(&self) -> f64 {
        // Weighted combination (epistemic most important for knowledge systems)
        0.5 * self.epistemic + 0.25 * self.aleatoric + 0.25 * self.structural
    }

    pub fn confidence(&self) -> f64 {
        1.0 - self.total()
    }
}

impl Default for Uncertainty3D {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }
}

/// Domain classification for queries
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDomain {
    Mathematics,
    Physics,
    History,
    Subjective,
    Technology,
    Environmental,
    Economic,
    Social,
    Medical,
    Political,
    Undefined,
    General,
}

impl std::fmt::Display for QueryDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// GIS classification result stored on-chain
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GisClassification {
    /// Unique classification ID
    pub id: String,
    /// Reference to the claim being classified
    pub claim_id: String,
    /// The query/statement being classified
    pub query: String,
    /// Type of ignorance detected
    pub ignorance_type: IgnoranceType,
    /// 3D uncertainty quantification
    pub uncertainty: Uncertainty3D,
    /// Domain of the query
    pub domain: QueryDomain,
    /// Expected Information Gain for resolution
    pub eig: f64,
    /// Whether this was published to Dark Spot DHT
    pub dark_spot_published: bool,
    /// Dark Spot signature hash (if published)
    pub dark_spot_signature: Option<String>,
    /// Classification timestamp
    pub classified_at: Timestamp,
    /// Agent who performed classification
    pub classifier: String,
}

/// The Eight Harmonies of the Kosmic Song
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Harmony {
    /// Resonant Coherence: Integration and wholeness
    ResonantCoherence,
    /// Pan-Sentient Flourishing: Care for all beings
    PanSentientFlourishing,
    /// Integral Wisdom: Truth and verification
    IntegralWisdom,
    /// Infinite Play: Creativity and possibility
    InfinitePlay,
    /// Universal Interconnectedness: Relational web
    UniversalInterconnectedness,
    /// Sacred Reciprocity: Balance and exchange
    SacredReciprocity,
    /// Evolutionary Progression: Growth and development
    EvolutionaryProgression,
}

impl Harmony {
    pub fn all() -> [Harmony; 7] {
        [
            Self::ResonantCoherence,
            Self::PanSentientFlourishing,
            Self::IntegralWisdom,
            Self::InfinitePlay,
            Self::UniversalInterconnectedness,
            Self::SacredReciprocity,
            Self::EvolutionaryProgression,
        ]
    }

    pub fn epistemic_mode(&self) -> &'static str {
        match self {
            Self::ResonantCoherence => "Integration-Knowing",
            Self::PanSentientFlourishing => "Care-Knowing",
            Self::IntegralWisdom => "Truth-Knowing",
            Self::InfinitePlay => "Play-Knowing",
            Self::UniversalInterconnectedness => "Web-Knowing",
            Self::SacredReciprocity => "Exchange-Knowing",
            Self::EvolutionaryProgression => "Growth-Knowing",
        }
    }

    pub fn primary_questions(&self) -> Vec<&'static str> {
        match self {
            Self::ResonantCoherence => vec![
                "How do the parts relate to each other?",
                "What patterns of integration exist?",
                "Where is there coherence or dissonance?",
            ],
            Self::PanSentientFlourishing => vec![
                "Who is affected by this situation?",
                "How does this impact well-being?",
                "Whose voice is not being heard?",
            ],
            Self::IntegralWisdom => vec![
                "What can be verified with certainty?",
                "What evidence supports each position?",
                "Where are the knowledge gaps?",
            ],
            Self::InfinitePlay => vec![
                "What creative possibilities exist?",
                "What novel approaches haven't been considered?",
                "Where is there room for experimentation?",
            ],
            Self::UniversalInterconnectedness => vec![
                "What connections exist between elements?",
                "What ripple effects might occur?",
                "Who and what are part of this web?",
            ],
            Self::SacredReciprocity => vec![
                "What flows back and forth?",
                "Is there balance in the exchange?",
                "Who gives and who receives?",
            ],
            Self::EvolutionaryProgression => vec![
                "What is emerging or developing?",
                "How has this evolved to this point?",
                "What trajectory are we on?",
            ],
        }
    }
}

impl std::fmt::Display for Harmony {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResonantCoherence => write!(f, "Resonant Coherence"),
            Self::PanSentientFlourishing => write!(f, "Pan-Sentient Flourishing"),
            Self::IntegralWisdom => write!(f, "Integral Wisdom"),
            Self::InfinitePlay => write!(f, "Infinite Play"),
            Self::UniversalInterconnectedness => write!(f, "Universal Interconnectedness"),
            Self::SacredReciprocity => write!(f, "Sacred Reciprocity"),
            Self::EvolutionaryProgression => write!(f, "Evolutionary Progression"),
        }
    }
}

/// A perspective generated from a Harmony frame
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonicPerspective {
    /// Which harmony generated this perspective
    pub harmony: Harmony,
    /// The perspective content
    pub content: String,
    /// Key insights from this perspective
    pub insights: Vec<String>,
    /// Concerns raised by this perspective
    pub concerns: Vec<String>,
    /// Confidence in this perspective (0.0 - 1.0)
    pub confidence: f64,
    /// Relevance to the situation (0.0 - 1.0)
    pub relevance: f64,
}

impl HarmonicPerspective {
    pub fn new(harmony: Harmony, content: &str) -> Self {
        Self {
            harmony,
            content: content.to_string(),
            insights: Vec::new(),
            concerns: Vec::new(),
            confidence: 0.5,
            relevance: 0.5,
        }
    }
}

/// A dissenting view preserved from synthesis
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PreservedDissent {
    /// Which harmony dissents
    pub harmony: Harmony,
    /// The dissenting content
    pub content: String,
    /// Why this dissent is important to preserve
    pub reason: String,
    /// How different from the unified view (0.0 - 1.0)
    pub divergence: f64,
}

/// Rashomon multi-perspective analysis result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RashomonAnalysis {
    /// Unique analysis ID
    pub id: String,
    /// Reference to the claim being analyzed
    pub claim_id: String,
    /// The situation/question being analyzed
    pub situation: String,
    /// Domain of the situation
    pub domain: QueryDomain,
    /// Perspectives from each relevant harmony
    pub perspectives: Vec<HarmonicPerspective>,
    /// Synthesized unified view
    pub unified_view: String,
    /// Points of agreement across harmonies
    pub agreements: Vec<String>,
    /// Preserved dissenting views
    pub preserved_dissents: Vec<PreservedDissent>,
    /// Overall synthesis confidence (0.0 - 1.0)
    pub synthesis_confidence: f64,
    /// Whether N3 (no-harm) boundaries were triggered
    pub n3_triggered: bool,
    /// Analysis timestamp
    pub analyzed_at: Timestamp,
    /// Agent who performed analysis
    pub analyzer: String,
}

/// Dark Spot record for collective ignorance tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DarkSpot {
    /// Unique dark spot ID
    pub id: String,
    /// The query representing the ignorance
    pub query: String,
    /// Type of ignorance
    pub ignorance_type: IgnoranceType,
    /// Domain of the ignorance
    pub domain: QueryDomain,
    /// Expected Information Gain
    pub eig: f64,
    /// Privacy-preserving signature hash
    pub zk_signature: String,
    /// Resolution status
    pub status: DarkSpotStatus,
    /// Resolution (if resolved)
    pub resolution: Option<String>,
    /// Resolver agent (if resolved)
    pub resolver: Option<String>,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Resolution timestamp (if resolved)
    pub resolved_at: Option<Timestamp>,
    /// Publishing agent
    pub publisher: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DarkSpotStatus {
    /// Actively seeking resolution
    Active,
    /// Resolution in progress
    ResolutionInProgress,
    /// Successfully resolved
    Resolved,
    /// Expired without resolution
    Expired,
    /// Determined to be impossible to resolve
    Impossible,
}

/// Harmonic alignment assessment for an agent or claim
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HarmonicAlignment {
    /// Unique alignment ID
    pub id: String,
    /// Subject (agent DID or claim ID)
    pub subject: String,
    /// Subject type
    pub subject_type: AlignmentSubjectType,
    /// Alignment scores for each harmony (0.0 - 1.0)
    pub harmony_scores: HarmonicScores,
    /// Overall alignment score (weighted average)
    pub overall_alignment: f64,
    /// Primary harmony (strongest alignment)
    pub primary_harmony: Harmony,
    /// Misaligned harmonies (score < 0.3)
    pub misalignments: Vec<Harmony>,
    /// Assessment timestamp
    pub assessed_at: Timestamp,
    /// Assessor agent
    pub assessor: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentSubjectType {
    Agent,
    Claim,
    Proposal,
    Action,
}

/// Scores for each of the Eight Harmonies
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonicScores {
    pub resonant_coherence: f64,
    pub pan_sentient_flourishing: f64,
    pub integral_wisdom: f64,
    pub infinite_play: f64,
    pub universal_interconnectedness: f64,
    pub sacred_reciprocity: f64,
    pub evolutionary_progression: f64,
}

impl HarmonicScores {
    pub fn new() -> Self {
        Self {
            resonant_coherence: 0.5,
            pan_sentient_flourishing: 0.5,
            integral_wisdom: 0.5,
            infinite_play: 0.5,
            universal_interconnectedness: 0.5,
            sacred_reciprocity: 0.5,
            evolutionary_progression: 0.5,
        }
    }

    pub fn get(&self, harmony: Harmony) -> f64 {
        match harmony {
            Harmony::ResonantCoherence => self.resonant_coherence,
            Harmony::PanSentientFlourishing => self.pan_sentient_flourishing,
            Harmony::IntegralWisdom => self.integral_wisdom,
            Harmony::InfinitePlay => self.infinite_play,
            Harmony::UniversalInterconnectedness => self.universal_interconnectedness,
            Harmony::SacredReciprocity => self.sacred_reciprocity,
            Harmony::EvolutionaryProgression => self.evolutionary_progression,
        }
    }

    pub fn set(&mut self, harmony: Harmony, value: f64) {
        let value = value.clamp(0.0, 1.0);
        match harmony {
            Harmony::ResonantCoherence => self.resonant_coherence = value,
            Harmony::PanSentientFlourishing => self.pan_sentient_flourishing = value,
            Harmony::IntegralWisdom => self.integral_wisdom = value,
            Harmony::InfinitePlay => self.infinite_play = value,
            Harmony::UniversalInterconnectedness => self.universal_interconnectedness = value,
            Harmony::SacredReciprocity => self.sacred_reciprocity = value,
            Harmony::EvolutionaryProgression => self.evolutionary_progression = value,
        }
    }

    pub fn weighted_average(&self) -> f64 {
        // Pan-Sentient Flourishing weighted highest (care ethics)
        let weights = [0.12, 0.20, 0.18, 0.10, 0.15, 0.13, 0.12];
        let scores = [
            self.resonant_coherence,
            self.pan_sentient_flourishing,
            self.integral_wisdom,
            self.infinite_play,
            self.universal_interconnectedness,
            self.sacred_reciprocity,
            self.evolutionary_progression,
        ];
        weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum()
    }

    pub fn primary_harmony(&self) -> Harmony {
        let mut max_harmony = Harmony::ResonantCoherence;
        let mut max_score = self.resonant_coherence;

        for harmony in Harmony::all() {
            let score = self.get(harmony);
            if score > max_score {
                max_score = score;
                max_harmony = harmony;
            }
        }
        max_harmony
    }

    pub fn misalignments(&self, threshold: f64) -> Vec<Harmony> {
        Harmony::all()
            .into_iter()
            .filter(|h| self.get(*h) < threshold)
            .collect()
    }
}

impl Default for HarmonicScores {
    fn default() -> Self {
        Self::new()
    }
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    KnowledgeQuery(KnowledgeQuery),
    EpistemicResult(EpistemicResult),
    ClaimReference(ClaimReference),
    KnowledgeBridgeEvent(KnowledgeBridgeEvent),
    // GIS v4.0 entry types
    GisClassification(GisClassification),
    RashomonAnalysis(RashomonAnalysis),
    DarkSpot(DarkSpot),
    HarmonicAlignment(HarmonicAlignment),
}

#[hdk_link_types]
pub enum LinkTypes {
    RecentQueries,
    SubjectToClaims,
    ClaimToEpistemic,
    RecentEvents,
    HappToClaims,
    // GIS v4.0 link types
    ClaimToGisClassification,
    ClaimToRashomonAnalysis,
    DomainToDarkSpots,
    AgentToHarmonicAlignment,
    HarmonyToClassifications,
    ActiveDarkSpots,
    ResolvedDarkSpots,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::KnowledgeQuery(query) => validate_create_query(action, query),
                EntryTypes::EpistemicResult(result) => validate_create_result(action, result),
                EntryTypes::ClaimReference(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::KnowledgeBridgeEvent(event) => validate_create_event(action, event),
                // GIS v4.0 validations
                EntryTypes::GisClassification(classification) => {
                    validate_create_gis_classification(action, classification)
                }
                EntryTypes::RashomonAnalysis(analysis) => {
                    validate_create_rashomon_analysis(action, analysis)
                }
                EntryTypes::DarkSpot(dark_spot) => {
                    validate_create_dark_spot(action, dark_spot)
                }
                EntryTypes::HarmonicAlignment(alignment) => {
                    validate_create_harmonic_alignment(action, alignment)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::KnowledgeQuery(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::EpistemicResult(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ClaimReference(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::KnowledgeBridgeEvent(_) => Ok(ValidateCallbackResult::Valid),
                // GIS entries: only DarkSpot can be updated (for resolution)
                EntryTypes::GisClassification(_) => Ok(ValidateCallbackResult::Invalid(
                    "GIS classifications are immutable".into(),
                )),
                EntryTypes::RashomonAnalysis(_) => Ok(ValidateCallbackResult::Invalid(
                    "Rashomon analyses are immutable".into(),
                )),
                EntryTypes::DarkSpot(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HarmonicAlignment(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::RecentQueries => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SubjectToClaims => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToEpistemic => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToClaims => Ok(ValidateCallbackResult::Valid),
            // GIS v4.0 link types
            LinkTypes::ClaimToGisClassification => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToRashomonAnalysis => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DomainToDarkSpots => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToHarmonicAlignment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HarmonyToClassifications => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveDarkSpots => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ResolvedDarkSpots => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete a link
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete a link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_query(_action: Create, query: KnowledgeQuery) -> ExternResult<ValidateCallbackResult> {
    if query.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Source hApp required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_result(_action: Create, result: EpistemicResult) -> ExternResult<ValidateCallbackResult> {
    if result.empirical < 0.0 || result.empirical > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Empirical must be 0.0-1.0".into()));
    }
    if result.normative < 0.0 || result.normative > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Normative must be 0.0-1.0".into()));
    }
    if result.mythic < 0.0 || result.mythic > 1.0 {
        return Ok(ValidateCallbackResult::Invalid("Mythic must be 0.0-1.0".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_event(_action: Create, event: KnowledgeBridgeEvent) -> ExternResult<ValidateCallbackResult> {
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Source hApp required".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// GIS v4.0 VALIDATION FUNCTIONS
// =============================================================================

fn validate_create_gis_classification(
    _action: Create,
    classification: GisClassification,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID format
    if classification.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "GIS classification ID required".into(),
        ));
    }

    // Validate query is present
    if classification.query.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query is required for classification".into(),
        ));
    }

    // Validate uncertainty values are in range
    if classification.uncertainty.epistemic < 0.0 || classification.uncertainty.epistemic > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Epistemic uncertainty must be 0.0-1.0".into(),
        ));
    }
    if classification.uncertainty.aleatoric < 0.0 || classification.uncertainty.aleatoric > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Aleatoric uncertainty must be 0.0-1.0".into(),
        ));
    }
    if classification.uncertainty.structural < 0.0 || classification.uncertainty.structural > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Structural uncertainty must be 0.0-1.0".into(),
        ));
    }

    // Validate EIG is in range
    if classification.eig < 0.0 || classification.eig > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Expected Information Gain must be 0.0-1.0".into(),
        ));
    }

    // Validate classifier is present
    if classification.classifier.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Classifier agent required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_rashomon_analysis(
    _action: Create,
    analysis: RashomonAnalysis,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID format
    if analysis.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rashomon analysis ID required".into(),
        ));
    }

    // Validate situation is present
    if analysis.situation.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Situation description required for analysis".into(),
        ));
    }

    // Validate at least one perspective
    if analysis.perspectives.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one harmonic perspective required".into(),
        ));
    }

    // Validate perspective confidences are in range
    for perspective in &analysis.perspectives {
        if perspective.confidence < 0.0 || perspective.confidence > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!(
                    "Perspective confidence for {} must be 0.0-1.0",
                    perspective.harmony
                ),
            ));
        }
        if perspective.relevance < 0.0 || perspective.relevance > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!(
                    "Perspective relevance for {} must be 0.0-1.0",
                    perspective.harmony
                ),
            ));
        }
    }

    // Validate synthesis confidence is in range
    if analysis.synthesis_confidence < 0.0 || analysis.synthesis_confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Synthesis confidence must be 0.0-1.0".into(),
        ));
    }

    // Validate dissent divergence values
    for dissent in &analysis.preserved_dissents {
        if dissent.divergence < 0.0 || dissent.divergence > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Dissent divergence for {} must be 0.0-1.0", dissent.harmony),
            ));
        }
    }

    // Validate analyzer is present
    if analysis.analyzer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Analyzer agent required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_dark_spot(
    _action: Create,
    dark_spot: DarkSpot,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID format
    if dark_spot.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dark spot ID required".into(),
        ));
    }

    // Validate query is present
    if dark_spot.query.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Query is required for dark spot".into(),
        ));
    }

    // Validate EIG is in range
    if dark_spot.eig < 0.0 || dark_spot.eig > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Expected Information Gain must be 0.0-1.0".into(),
        ));
    }

    // Validate ZK signature is present
    if dark_spot.zk_signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ZK signature required for privacy preservation".into(),
        ));
    }

    // Validate publisher is present
    if dark_spot.publisher.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Publisher agent required".into(),
        ));
    }

    // Validate initial status is Active
    if dark_spot.status != DarkSpotStatus::Active {
        return Ok(ValidateCallbackResult::Invalid(
            "New dark spots must have Active status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_harmonic_alignment(
    _action: Create,
    alignment: HarmonicAlignment,
) -> ExternResult<ValidateCallbackResult> {
    // Validate ID format
    if alignment.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Harmonic alignment ID required".into(),
        ));
    }

    // Validate subject is present
    if alignment.subject.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject (agent DID or claim ID) required".into(),
        ));
    }

    // Validate harmony scores are in range
    for harmony in Harmony::all() {
        let score = alignment.harmony_scores.get(harmony);
        if score < 0.0 || score > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Harmony score for {} must be 0.0-1.0", harmony),
            ));
        }
    }

    // Validate overall alignment is in range
    if alignment.overall_alignment < 0.0 || alignment.overall_alignment > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Overall alignment must be 0.0-1.0".into(),
        ));
    }

    // Validate assessor is present
    if alignment.assessor.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Assessor agent required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// UNIT TESTS FOR GIS v4.0 TYPES
// =============================================================================

#[cfg(test)]
mod gis_classification_tests {
    use super::*;

    #[test]
    fn test_ignorance_type_symbols() {
        assert_eq!(IgnoranceType::Known.symbol(), "κ");
        assert_eq!(IgnoranceType::KnownUnknown.symbol(), "ι₁");
        assert_eq!(IgnoranceType::UnknownUnknown.symbol(), "ι₂");
        assert_eq!(IgnoranceType::StructuralUnknown.symbol(), "ι₃");
        assert_eq!(IgnoranceType::Impossible.symbol(), "ι∞");
    }

    #[test]
    fn test_ignorance_type_resolvability() {
        assert_eq!(IgnoranceType::Known.resolvability(), 1.0);
        assert_eq!(IgnoranceType::KnownUnknown.resolvability(), 0.7);
        assert_eq!(IgnoranceType::UnknownUnknown.resolvability(), 0.3);
        assert_eq!(IgnoranceType::StructuralUnknown.resolvability(), 0.1);
        assert_eq!(IgnoranceType::Impossible.resolvability(), 0.0);
    }

    #[test]
    fn test_uncertainty_3d_creation() {
        let uncertainty = Uncertainty3D::new(0.5, 0.3, 0.2);
        assert_eq!(uncertainty.epistemic, 0.5);
        assert_eq!(uncertainty.aleatoric, 0.3);
        assert_eq!(uncertainty.structural, 0.2);
    }

    #[test]
    fn test_uncertainty_3d_clamping() {
        let uncertainty = Uncertainty3D::new(1.5, -0.5, 2.0);
        assert_eq!(uncertainty.epistemic, 1.0);
        assert_eq!(uncertainty.aleatoric, 0.0);
        assert_eq!(uncertainty.structural, 1.0);
    }

    #[test]
    fn test_uncertainty_3d_total() {
        let uncertainty = Uncertainty3D::new(0.4, 0.2, 0.2);
        let total = uncertainty.total();
        assert!((total - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_uncertainty_3d_confidence() {
        let uncertainty = Uncertainty3D::new(0.4, 0.2, 0.2);
        let confidence = uncertainty.confidence();
        assert!((confidence - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_query_domain_display() {
        assert_eq!(format!("{}", QueryDomain::Mathematics), "Mathematics");
        assert_eq!(format!("{}", QueryDomain::Environmental), "Environmental");
    }
}

#[cfg(test)]
mod harmony_tests {
    use super::*;

    #[test]
    fn test_harmony_all() {
        let harmonies = Harmony::all();
        assert_eq!(harmonies.len(), 7);
        assert!(harmonies.contains(&Harmony::ResonantCoherence));
        assert!(harmonies.contains(&Harmony::PanSentientFlourishing));
        assert!(harmonies.contains(&Harmony::IntegralWisdom));
        assert!(harmonies.contains(&Harmony::InfinitePlay));
        assert!(harmonies.contains(&Harmony::UniversalInterconnectedness));
        assert!(harmonies.contains(&Harmony::SacredReciprocity));
        assert!(harmonies.contains(&Harmony::EvolutionaryProgression));
    }

    #[test]
    fn test_harmony_epistemic_modes() {
        assert_eq!(Harmony::ResonantCoherence.epistemic_mode(), "Integration-Knowing");
        assert_eq!(Harmony::PanSentientFlourishing.epistemic_mode(), "Care-Knowing");
        assert_eq!(Harmony::IntegralWisdom.epistemic_mode(), "Truth-Knowing");
        assert_eq!(Harmony::InfinitePlay.epistemic_mode(), "Play-Knowing");
        assert_eq!(Harmony::UniversalInterconnectedness.epistemic_mode(), "Web-Knowing");
        assert_eq!(Harmony::SacredReciprocity.epistemic_mode(), "Exchange-Knowing");
        assert_eq!(Harmony::EvolutionaryProgression.epistemic_mode(), "Growth-Knowing");
    }

    #[test]
    fn test_harmony_primary_questions() {
        let questions = Harmony::PanSentientFlourishing.primary_questions();
        assert!(!questions.is_empty());
        assert!(questions.iter().any(|q| q.contains("affected")));
    }

    #[test]
    fn test_harmony_display() {
        assert_eq!(format!("{}", Harmony::PanSentientFlourishing), "Pan-Sentient Flourishing");
        assert_eq!(format!("{}", Harmony::IntegralWisdom), "Integral Wisdom");
    }
}

#[cfg(test)]
mod harmonic_scores_tests {
    use super::*;

    #[test]
    fn test_harmonic_scores_new() {
        let scores = HarmonicScores::new();
        assert_eq!(scores.resonant_coherence, 0.5);
        assert_eq!(scores.pan_sentient_flourishing, 0.5);
        assert_eq!(scores.integral_wisdom, 0.5);
    }

    #[test]
    fn test_harmonic_scores_get_set() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.9);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 0.9);
    }

    #[test]
    fn test_harmonic_scores_clamping() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 1.5);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 1.0);

        scores.set(Harmony::IntegralWisdom, -0.5);
        assert_eq!(scores.get(Harmony::IntegralWisdom), 0.0);
    }

    #[test]
    fn test_harmonic_scores_weighted_average() {
        let mut scores = HarmonicScores::new();
        for harmony in Harmony::all() {
            scores.set(harmony, 1.0);
        }
        let avg = scores.weighted_average();
        assert!((avg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_harmonic_scores_primary_harmony() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.9);
        scores.set(Harmony::PanSentientFlourishing, 0.7);
        assert_eq!(scores.primary_harmony(), Harmony::IntegralWisdom);
    }

    #[test]
    fn test_harmonic_scores_misalignments() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::IntegralWisdom, 0.2);
        scores.set(Harmony::InfinitePlay, 0.1);

        let misaligned = scores.misalignments(0.3);
        assert!(misaligned.contains(&Harmony::IntegralWisdom));
        assert!(misaligned.contains(&Harmony::InfinitePlay));
    }
}

#[cfg(test)]
mod perspective_tests {
    use super::*;

    #[test]
    fn test_harmonic_perspective_new() {
        let perspective = HarmonicPerspective::new(
            Harmony::IntegralWisdom,
            "Truth-seeking perspective on the matter"
        );
        assert_eq!(perspective.harmony, Harmony::IntegralWisdom);
        assert!(perspective.content.contains("Truth-seeking"));
        assert_eq!(perspective.confidence, 0.5);
        assert_eq!(perspective.relevance, 0.5);
    }
}

#[cfg(test)]
mod dark_spot_tests {
    use super::*;

    #[test]
    fn test_dark_spot_status_values() {
        assert_ne!(DarkSpotStatus::Active, DarkSpotStatus::Resolved);
        assert_ne!(DarkSpotStatus::ResolutionInProgress, DarkSpotStatus::Expired);
        assert_ne!(DarkSpotStatus::Impossible, DarkSpotStatus::Active);
    }
}

#[cfg(test)]
mod workflow_tests {
    use super::*;

    #[test]
    fn test_gis_to_dark_spot_workflow() {
        let ignorance_type = IgnoranceType::KnownUnknown;
        let uncertainty = Uncertainty3D::new(0.5, 0.3, 0.2);

        let potential_value = match ignorance_type {
            IgnoranceType::Known => 0.9,
            IgnoranceType::KnownUnknown => 0.7,
            IgnoranceType::UnknownUnknown => 0.4,
            IgnoranceType::StructuralUnknown => 0.2,
            IgnoranceType::Impossible => 0.05,
        };
        let likelihood = 1.0 - uncertainty.epistemic;
        let eig = potential_value * likelihood;

        assert!(eig >= 0.3, "EIG {} should meet Dark Spot threshold", eig);

        let status = DarkSpotStatus::Active;
        assert_eq!(status, DarkSpotStatus::Active);
    }

    #[test]
    fn test_rashomon_synthesis_workflow() {
        let mut perspectives = Vec::new();

        for harmony in &[Harmony::PanSentientFlourishing, Harmony::IntegralWisdom, Harmony::SacredReciprocity] {
            let mut perspective = HarmonicPerspective::new(
                *harmony,
                &format!("{} perspective on the situation", harmony)
            );
            perspective.relevance = 0.8;
            perspective.confidence = 0.7;
            perspectives.push(perspective);
        }

        let avg_confidence: f64 = perspectives.iter().map(|p| p.confidence).sum::<f64>()
            / perspectives.len() as f64;
        let avg_relevance: f64 = perspectives.iter().map(|p| p.relevance).sum::<f64>()
            / perspectives.len() as f64;
        let synthesis_confidence = (avg_confidence * 0.6 + avg_relevance * 0.4).min(1.0);

        assert!(synthesis_confidence > 0.7, "Synthesis confidence should be high");
    }

    #[test]
    fn test_harmonic_alignment_workflow() {
        let mut scores = HarmonicScores::new();
        scores.set(Harmony::PanSentientFlourishing, 0.9);
        scores.set(Harmony::IntegralWisdom, 0.8);
        scores.set(Harmony::SacredReciprocity, 0.7);
        scores.set(Harmony::ResonantCoherence, 0.6);

        let overall = scores.weighted_average();
        assert!(overall > 0.6, "Overall alignment should be good");

        let primary = scores.primary_harmony();
        assert_eq!(primary, Harmony::PanSentientFlourishing);

        let misalignments = scores.misalignments(0.3);
        assert!(misalignments.is_empty(), "Should have no misalignments");
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_uncertainty() {
        let uncertainty = Uncertainty3D::new(0.0, 0.0, 0.0);
        assert_eq!(uncertainty.total(), 0.0);
        assert_eq!(uncertainty.confidence(), 1.0);
    }

    #[test]
    fn test_max_uncertainty() {
        let uncertainty = Uncertainty3D::new(1.0, 1.0, 1.0);
        assert_eq!(uncertainty.total(), 1.0);
        assert_eq!(uncertainty.confidence(), 0.0);
    }

    #[test]
    fn test_all_harmonies_zero() {
        let mut scores = HarmonicScores::new();
        for harmony in Harmony::all() {
            scores.set(harmony, 0.0);
        }
        assert_eq!(scores.weighted_average(), 0.0);
    }

    #[test]
    fn test_misalignments_with_zero_threshold() {
        let scores = HarmonicScores::new();
        let misalignments = scores.misalignments(0.0);
        assert!(misalignments.is_empty(), "No harmony should be below 0.0");
    }

    #[test]
    fn test_misalignments_with_high_threshold() {
        let scores = HarmonicScores::new();
        let misalignments = scores.misalignments(0.6);
        assert_eq!(misalignments.len(), 7);
    }
}
