/// Stable, root-independent view of the parts of StructuredThought that a text
/// renderer may consume. The root `symthaea` crate can map into this type
/// without this bridge depending back on the root crate.
///
/// The contract intentionally preserves the cognitive state that the existing
/// StructuredThought text prompt exposes while keeping executable/code-bearing
/// state out of the heterogeneous text path.
#[derive(Clone, Debug, PartialEq)]
pub struct BrocaTranslationPlan {
    pub intent: RendererIntent,
    pub response_type: RendererResponseType,
    pub epistemic_status: RendererEpistemicStatus,

    /// Ψ — Symthaea's composite consciousness estimate (not IIT Phi).
    pub psi: f64,
    pub meta_awareness: f64,
    pub coherence: f64,

    /// Affective state used to shape rendering without changing factual content.
    pub valence: f64,
    pub arousal: f64,
    pub warmth: f64,

    /// Typed relational context. These values are control-safe because the
    /// lower contract does not carry arbitrary strings for relationship state.
    pub relationship_stage: BrocaRelationshipStage,
    pub relation_mode: BrocaRelationMode,
    pub trust: f32,

    pub activated_concepts: Vec<BrocaConcept>,
    pub structured_data: Option<BrocaStructuredData>,
    pub domain_context: Option<BrocaDomainContext>,
    pub constraints: Vec<BrocaConstraint>,
    pub original_input: Option<String>,

    /// Ontological tiers are semantic context only, never executable commands.
    pub primitive_tiers: Vec<String>,

    /// Code-bearing plans stay on the existing code/native translation path.
    pub code_bearing: bool,
}

impl Default for BrocaTranslationPlan {
    fn default() -> Self {
        Self {
            intent: RendererIntent::Unknown,
            response_type: RendererResponseType::Statement,
            epistemic_status: RendererEpistemicStatus::Unknown,
            psi: 0.0,
            meta_awareness: 0.0,
            coherence: 0.0,
            valence: 0.0,
            arousal: 0.0,
            warmth: 0.5,
            relationship_stage: BrocaRelationshipStage::NoRelation,
            relation_mode: BrocaRelationMode::IIt,
            trust: 0.0,
            activated_concepts: vec![],
            structured_data: None,
            domain_context: None,
            constraints: vec![],
            original_input: None,
            primitive_tiers: vec![],
            code_bearing: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaConcept {
    pub name: String,
    pub activation: f32,
    pub relevance: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BrocaStructuredData {
    List(Vec<String>),
    KeyValue(Vec<(String, String)>),
    Numeric {
        value: f64,
        unit: Option<String>,
    },
    /// Marker only. Code itself remains on the existing code path.
    Code,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaDomainContext {
    pub domain: String,
    pub entities: Vec<BrocaEntity>,
    pub computed_answer: Option<String>,
    pub epistemic_cube: Option<BrocaEpistemicCube>,
    pub psi: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaEntity {
    pub entity_type: String,
    pub value: String,
    pub confidence: f64,
}

/// Root-independent form of Symthaea's E/N/M/H epistemic classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrocaEpistemicCube {
    /// Empirical tier: 0..=4.
    pub empirical: u8,
    /// Normative tier: 0..=3.
    pub normative: u8,
    /// Materiality tier: 0..=3.
    pub materiality: u8,
    /// Optional harmonic tier: 0..=4.
    pub harmonic: Option<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaConstraint {
    pub kind: BrocaConstraintKind,
    /// Legacy free-form constraint text is audit data only. It never becomes
    /// trusted renderer control. The hardened adapter can refuse to claim
    /// faithful translation when this text is required but not safely typed.
    pub audit_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrocaConstraintKind {
    MaxLength,
    Tone,
    MustInclude,
    MustExclude,
    Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererIntent {
    Acknowledge,
    Answer,
    Clarify,
    ProposeAction,
    ExpressUncertainty,
    Reflect,
    Continue,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererResponseType {
    Greeting,
    Statement,
    Question,
    ActionConfirmation,
    Report,
    Empathic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererEpistemicStatus {
    Certain,
    Probable,
    Uncertain,
    Unknown,
    OutOfDomain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererTone {
    Neutral,
    Natural,
    Warm,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrocaRelationshipStage {
    #[default]
    NoRelation,
    Awareness,
    Contact,
    Attunement,
    Bonding,
    Unity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrocaRelationMode {
    #[default]
    IIt,
    IThou,
}
