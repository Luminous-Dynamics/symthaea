/// Stable, root-independent view of the parts of StructuredThought that a text
/// renderer may consume. The root `symthaea` crate can map into this type
/// without this bridge depending back on the root crate.
#[derive(Clone, Debug, PartialEq)]
pub struct BrocaTranslationPlan {
    pub intent: RendererIntent,
    pub response_type: RendererResponseType,
    pub epistemic_status: RendererEpistemicStatus,
    pub warmth: f64,
    pub meta_awareness: f64,
    pub coherence: f64,
    pub activated_concepts: Vec<BrocaConcept>,
    pub structured_data: Option<BrocaStructuredData>,
    pub domain_context: Option<BrocaDomainContext>,
    pub constraints: Vec<BrocaConstraint>,
    pub original_input: Option<String>,
    /// Code-bearing plans stay on the existing code/native translation path.
    pub code_bearing: bool,
}

impl Default for BrocaTranslationPlan {
    fn default() -> Self {
        Self {
            intent: RendererIntent::Unknown,
            response_type: RendererResponseType::Statement,
            epistemic_status: RendererEpistemicStatus::Unknown,
            warmth: 0.5,
            meta_awareness: 0.0,
            coherence: 0.0,
            activated_concepts: vec![],
            structured_data: None,
            domain_context: None,
            constraints: vec![],
            original_input: None,
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaEntity {
    pub entity_type: String,
    pub value: String,
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrocaConstraint {
    pub kind: BrocaConstraintKind,
    /// Free-form text is audit data only. It never becomes trusted renderer control.
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
