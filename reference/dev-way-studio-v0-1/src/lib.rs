#![forbid(unsafe_code)]

//! Pure reference semantics for DEV-WAY Studio candidate generation.
//!
//! This crate intentionally owns no model, filesystem, process, network,
//! persistence, verifier, source-mutation, or effect capability.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const MAX_ID_LEN: usize = 128;
const MAX_PROSE_LEN: usize = 4096;
const MAX_ITEM_LEN: usize = 1024;
const MAX_ITEMS: usize = 64;
const MAX_CANDIDATES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StudioError {
    EmptyField { field: &'static str },
    FieldTooLong { field: &'static str, max: usize },
    TooManyItems { field: &'static str, max: usize },
    NonCanonicalText { field: &'static str },
    ControlCharacter { field: &'static str },
    MissingInvariant,
    ParentIntentMismatch,
    MissingDivergenceAxis,
    DuplicateCandidateConflict { candidate_id: String },
    CandidateLimitExceeded { max: usize },
}

impl fmt::Display for StudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(f, "field `{field}` is empty"),
            Self::FieldTooLong { field, max } => {
                write!(f, "field `{field}` exceeds maximum length {max}")
            }
            Self::TooManyItems { field, max } => {
                write!(f, "field `{field}` exceeds maximum item count {max}")
            }
            Self::NonCanonicalText { field } => {
                write!(f, "field `{field}` is not in canonical text form")
            }
            Self::ControlCharacter { field } => {
                write!(f, "field `{field}` contains control characters")
            }
            Self::MissingInvariant => write!(f, "creative intent requires at least one invariant"),
            Self::ParentIntentMismatch => write!(f, "candidate parent intent does not match"),
            Self::MissingDivergenceAxis => {
                write!(f, "creative candidate requires at least one divergence axis")
            }
            Self::DuplicateCandidateConflict { candidate_id } => write!(
                f,
                "candidate id `{candidate_id}` is reused for conflicting candidate content"
            ),
            Self::CandidateLimitExceeded { max } => {
                write!(f, "candidate collection exceeds maximum size {max}")
            }
        }
    }
}

impl Error for StudioError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeIntentDraftV1 {
    pub intent_id: String,
    pub desired_experience: String,
    pub invariants: Vec<String>,
    pub freedoms: Vec<String>,
    pub non_goals: Vec<String>,
    pub taste_terms: Vec<String>,
    pub examples: Vec<String>,
    pub anti_examples: Vec<String>,
    pub complexity_budget: Option<String>,
    pub reversibility_expectation: Option<String>,
    pub unresolved_questions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeIntentV1 {
    intent_id: String,
    desired_experience: String,
    invariants: Vec<String>,
    freedoms: Vec<String>,
    non_goals: Vec<String>,
    taste_terms: Vec<String>,
    examples: Vec<String>,
    anti_examples: Vec<String>,
    complexity_budget: Option<String>,
    reversibility_expectation: Option<String>,
    unresolved_questions: Vec<String>,
}

impl CreativeIntentV1 {
    pub fn intent_id(&self) -> &str {
        &self.intent_id
    }

    pub fn desired_experience(&self) -> &str {
        &self.desired_experience
    }

    pub fn invariants(&self) -> &[String] {
        &self.invariants
    }

    pub fn freedoms(&self) -> &[String] {
        &self.freedoms
    }

    pub fn non_goals(&self) -> &[String] {
        &self.non_goals
    }

    pub fn taste_terms(&self) -> &[String] {
        &self.taste_terms
    }

    pub fn examples(&self) -> &[String] {
        &self.examples
    }

    pub fn anti_examples(&self) -> &[String] {
        &self.anti_examples
    }

    pub fn complexity_budget(&self) -> Option<&str> {
        self.complexity_budget.as_deref()
    }

    pub fn reversibility_expectation(&self) -> Option<&str> {
        self.reversibility_expectation.as_deref()
    }

    pub fn unresolved_questions(&self) -> &[String] {
        &self.unresolved_questions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DivergenceAxisV1 {
    DataModel,
    InteractionModel,
    ControlFlow,
    ArchitectureScale,
    StateExplicitness,
    DistributionAssumption,
    DependencyStrategy,
    ApiShape,
    UserMetaphor,
    MinimalVsExpressive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeCandidateDraftV1 {
    pub candidate_id: String,
    pub parent_intent_id: String,
    /// Declared ancestry/family label only. It is not proof of material design diversity.
    pub family_id: String,
    pub divergence_axes: Vec<DivergenceAxisV1>,
    pub summary: String,
    pub tradeoffs: Vec<String>,
    pub assumptions: Vec<String>,
    pub subtraction_notes: Vec<String>,
    pub unresolved_risks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeCandidateV1 {
    candidate_id: String,
    parent_intent_id: String,
    family_id: String,
    divergence_axes: Vec<DivergenceAxisV1>,
    summary: String,
    tradeoffs: Vec<String>,
    assumptions: Vec<String>,
    subtraction_notes: Vec<String>,
    unresolved_risks: Vec<String>,
}

impl CreativeCandidateV1 {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn parent_intent_id(&self) -> &str {
        &self.parent_intent_id
    }

    /// Declared ancestry/family label; not evidence that this candidate is materially independent.
    pub fn family_id(&self) -> &str {
        &self.family_id
    }

    pub fn divergence_axes(&self) -> &[DivergenceAxisV1] {
        &self.divergence_axes
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }

    pub fn tradeoffs(&self) -> &[String] {
        &self.tradeoffs
    }

    pub fn assumptions(&self) -> &[String] {
        &self.assumptions
    }

    pub fn subtraction_notes(&self) -> &[String] {
        &self.subtraction_notes
    }

    pub fn unresolved_risks(&self) -> &[String] {
        &self.unresolved_risks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CreativeDispositionV1 {
    KeepExploring,
    PromisingCandidate,
    TooComplex,
    TechnicallyGoodWrongFeel,
    ElegantButUnproven,
    SelectedForSpecification,
    RejectedForNow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreativeDispositionRecordV1 {
    candidate_id: String,
    disposition: CreativeDispositionV1,
    human_rationale: Option<String>,
}

impl CreativeDispositionRecordV1 {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn disposition(&self) -> CreativeDispositionV1 {
        self.disposition
    }

    pub fn human_rationale(&self) -> Option<&str> {
        self.human_rationale.as_deref()
    }

    /// Candidate-workflow selection only. This does not verify code or grant authority.
    pub fn selected_for_specification(&self) -> bool {
        self.disposition == CreativeDispositionV1::SelectedForSpecification
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredFamilyCountV1 {
    family_id: String,
    candidate_count: usize,
}

impl DeclaredFamilyCountV1 {
    pub fn family_id(&self) -> &str {
        &self.family_id
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_count
    }
}

pub fn admit_intent(draft: CreativeIntentDraftV1) -> Result<CreativeIntentV1, StudioError> {
    let intent_id = canonical_identity("intent_id", draft.intent_id)?;
    let desired_experience = canonical_required_prose(
        "desired_experience",
        draft.desired_experience,
        MAX_PROSE_LEN,
    )?;
    let invariants = canonical_string_set("invariants", draft.invariants, false)?;
    if invariants.is_empty() {
        return Err(StudioError::MissingInvariant);
    }

    Ok(CreativeIntentV1 {
        intent_id,
        desired_experience,
        invariants,
        freedoms: canonical_string_set("freedoms", draft.freedoms, false)?,
        non_goals: canonical_string_set("non_goals", draft.non_goals, false)?,
        taste_terms: canonical_string_set("taste_terms", draft.taste_terms, false)?,
        examples: canonical_string_set("examples", draft.examples, false)?,
        anti_examples: canonical_string_set("anti_examples", draft.anti_examples, false)?,
        complexity_budget: canonical_optional_prose(
            "complexity_budget",
            draft.complexity_budget,
            MAX_ITEM_LEN,
        )?,
        reversibility_expectation: canonical_optional_prose(
            "reversibility_expectation",
            draft.reversibility_expectation,
            MAX_ITEM_LEN,
        )?,
        unresolved_questions: canonical_string_set(
            "unresolved_questions",
            draft.unresolved_questions,
            false,
        )?,
    })
}

pub fn admit_candidate(
    intent: &CreativeIntentV1,
    draft: CreativeCandidateDraftV1,
) -> Result<CreativeCandidateV1, StudioError> {
    let candidate_id = canonical_identity("candidate_id", draft.candidate_id)?;
    let parent_intent_id = canonical_identity("parent_intent_id", draft.parent_intent_id)?;
    if parent_intent_id != intent.intent_id {
        return Err(StudioError::ParentIntentMismatch);
    }
    let family_id = canonical_identity("family_id", draft.family_id)?;

    if draft.divergence_axes.len() > MAX_ITEMS {
        return Err(StudioError::TooManyItems {
            field: "divergence_axes",
            max: MAX_ITEMS,
        });
    }
    let divergence_axes: Vec<_> = draft
        .divergence_axes
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if divergence_axes.is_empty() {
        return Err(StudioError::MissingDivergenceAxis);
    }

    Ok(CreativeCandidateV1 {
        candidate_id,
        parent_intent_id,
        family_id,
        divergence_axes,
        summary: canonical_required_prose("summary", draft.summary, MAX_PROSE_LEN)?,
        tradeoffs: canonical_string_set("tradeoffs", draft.tradeoffs, false)?,
        assumptions: canonical_string_set("assumptions", draft.assumptions, false)?,
        subtraction_notes: canonical_string_set(
            "subtraction_notes",
            draft.subtraction_notes,
            false,
        )?,
        unresolved_risks: canonical_string_set(
            "unresolved_risks",
            draft.unresolved_risks,
            false,
        )?,
    })
}

pub fn record_disposition(
    candidate: &CreativeCandidateV1,
    disposition: CreativeDispositionV1,
    human_rationale: Option<String>,
) -> Result<CreativeDispositionRecordV1, StudioError> {
    Ok(CreativeDispositionRecordV1 {
        candidate_id: candidate.candidate_id.clone(),
        disposition,
        human_rationale: canonical_optional_prose(
            "human_rationale",
            human_rationale,
            MAX_PROSE_LEN,
        )?,
    })
}

pub fn canonicalize_candidates(
    intent: &CreativeIntentV1,
    drafts: impl IntoIterator<Item = CreativeCandidateDraftV1>,
) -> Result<Vec<CreativeCandidateV1>, StudioError> {
    let mut by_id = BTreeMap::<String, CreativeCandidateV1>::new();
    let mut observed = 0usize;

    for draft in drafts {
        observed = observed.saturating_add(1);
        if observed > MAX_CANDIDATES {
            return Err(StudioError::CandidateLimitExceeded {
                max: MAX_CANDIDATES,
            });
        }

        let candidate = admit_candidate(intent, draft)?;
        match by_id.get(candidate.candidate_id()) {
            Some(existing) if existing == &candidate => {}
            Some(_) => {
                return Err(StudioError::DuplicateCandidateConflict {
                    candidate_id: candidate.candidate_id.clone(),
                });
            }
            None => {
                by_id.insert(candidate.candidate_id.clone(), candidate);
            }
        }
    }

    Ok(by_id.into_values().collect())
}

/// Census of caller/workflow-declared ancestry families.
///
/// This intentionally does not claim that distinct family IDs prove materially
/// distinct designs. The paired benchmark owns that stronger assessment.
pub fn declared_family_census(candidates: &[CreativeCandidateV1]) -> Vec<DeclaredFamilyCountV1> {
    let mut counts = BTreeMap::<String, usize>::new();
    for candidate in candidates {
        *counts.entry(candidate.family_id.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(family_id, candidate_count)| DeclaredFamilyCountV1 {
            family_id,
            candidate_count,
        })
        .collect()
}

pub fn declared_family_count(candidates: &[CreativeCandidateV1]) -> usize {
    declared_family_census(candidates).len()
}

fn canonical_identity(field: &'static str, value: String) -> Result<String, StudioError> {
    canonical_required_prose(field, value, MAX_ID_LEN)
}

fn canonical_required_prose(
    field: &'static str,
    value: String,
    max: usize,
) -> Result<String, StudioError> {
    if value.is_empty() {
        return Err(StudioError::EmptyField { field });
    }
    if value.len() > max {
        return Err(StudioError::FieldTooLong { field, max });
    }
    if value.trim() != value {
        return Err(StudioError::NonCanonicalText { field });
    }
    if value.chars().any(char::is_control) {
        return Err(StudioError::ControlCharacter { field });
    }
    Ok(value)
}

fn canonical_optional_prose(
    field: &'static str,
    value: Option<String>,
    max: usize,
) -> Result<Option<String>, StudioError> {
    value
        .map(|value| canonical_required_prose(field, value, max))
        .transpose()
}

fn canonical_string_set(
    field: &'static str,
    values: Vec<String>,
    required: bool,
) -> Result<Vec<String>, StudioError> {
    if values.len() > MAX_ITEMS {
        return Err(StudioError::TooManyItems {
            field,
            max: MAX_ITEMS,
        });
    }

    let mut result = BTreeSet::new();
    for value in values {
        result.insert(canonical_required_prose(field, value, MAX_ITEM_LEN)?);
    }

    if required && result.is_empty() {
        return Err(StudioError::EmptyField { field });
    }
    Ok(result.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent_draft() -> CreativeIntentDraftV1 {
        CreativeIntentDraftV1 {
            intent_id: "intent-001".into(),
            desired_experience: "Make the workflow feel obvious without hiding state.".into(),
            invariants: vec!["State transitions remain explicit.".into()],
            freedoms: vec!["The UI structure may change.".into()],
            non_goals: vec!["Do not add network behavior.".into()],
            taste_terms: vec!["quiet".into(), "legible".into()],
            examples: vec!["small local demo".into()],
            anti_examples: vec!["wizard with hidden steps".into()],
            complexity_budget: Some("Prefer one small module.".into()),
            reversibility_expectation: Some("Prototype must be removable.".into()),
            unresolved_questions: vec!["Should history be visible by default?".into()],
        }
    }

    fn candidate_draft(id: &str, family: &str) -> CreativeCandidateDraftV1 {
        CreativeCandidateDraftV1 {
            candidate_id: id.into(),
            parent_intent_id: "intent-001".into(),
            family_id: family.into(),
            divergence_axes: vec![DivergenceAxisV1::InteractionModel],
            summary: format!("Candidate {id}"),
            tradeoffs: vec!["More explicit state for slightly more UI.".into()],
            assumptions: vec!["Local interaction only.".into()],
            subtraction_notes: vec!["No external service.".into()],
            unresolved_risks: vec!["May be too verbose for tiny screens.".into()],
        }
    }

    #[test]
    fn valid_intent_is_admitted() {
        let intent = admit_intent(intent_draft()).unwrap();
        assert_eq!(intent.intent_id(), "intent-001");
        assert_eq!(intent.invariants().len(), 1);
    }

    #[test]
    fn blank_desired_experience_is_rejected() {
        let mut draft = intent_draft();
        draft.desired_experience.clear();
        assert_eq!(
            admit_intent(draft),
            Err(StudioError::EmptyField {
                field: "desired_experience"
            })
        );
    }

    #[test]
    fn intent_without_invariants_is_rejected() {
        let mut draft = intent_draft();
        draft.invariants.clear();
        assert_eq!(admit_intent(draft), Err(StudioError::MissingInvariant));
    }

    #[test]
    fn noncanonical_identity_is_rejected() {
        let mut draft = intent_draft();
        draft.intent_id = " intent-001".into();
        assert_eq!(
            admit_intent(draft),
            Err(StudioError::NonCanonicalText { field: "intent_id" })
        );
    }

    #[test]
    fn candidate_parent_mismatch_is_rejected() {
        let intent = admit_intent(intent_draft()).unwrap();
        let mut draft = candidate_draft("c1", "family-a");
        draft.parent_intent_id = "another-intent".into();
        assert_eq!(
            admit_candidate(&intent, draft),
            Err(StudioError::ParentIntentMismatch)
        );
    }

    #[test]
    fn candidate_requires_meaningful_divergence_axis() {
        let intent = admit_intent(intent_draft()).unwrap();
        let mut draft = candidate_draft("c1", "family-a");
        draft.divergence_axes.clear();
        assert_eq!(
            admit_candidate(&intent, draft),
            Err(StudioError::MissingDivergenceAxis)
        );
    }

    #[test]
    fn duplicate_axes_normalize_deterministically() {
        let intent = admit_intent(intent_draft()).unwrap();
        let mut draft = candidate_draft("c1", "family-a");
        draft.divergence_axes = vec![
            DivergenceAxisV1::ApiShape,
            DivergenceAxisV1::InteractionModel,
            DivergenceAxisV1::ApiShape,
        ];
        let candidate = admit_candidate(&intent, draft).unwrap();
        assert_eq!(
            candidate.divergence_axes(),
            &[DivergenceAxisV1::InteractionModel, DivergenceAxisV1::ApiShape]
        );
    }

    #[test]
    fn set_like_fields_normalize_deterministically() {
        let mut draft = intent_draft();
        draft.taste_terms = vec!["quiet".into(), "legible".into(), "quiet".into()];
        let intent = admit_intent(draft).unwrap();
        assert_eq!(
            intent.taste_terms(),
            &[String::from("legible"), String::from("quiet")]
        );
    }

    #[test]
    fn five_descendants_of_one_declared_family_count_as_one_declared_family() {
        let intent = admit_intent(intent_draft()).unwrap();
        let drafts = (0..5).map(|index| candidate_draft(&format!("c{index}"), "family-a"));
        let candidates = canonicalize_candidates(&intent, drafts).unwrap();
        assert_eq!(candidates.len(), 5);
        assert_eq!(declared_family_count(&candidates), 1);
        assert_eq!(declared_family_census(&candidates)[0].candidate_count(), 5);
    }

    #[test]
    fn distinct_declared_families_remain_distinct_even_with_similar_summaries() {
        let intent = admit_intent(intent_draft()).unwrap();
        let mut a = candidate_draft("c1", "family-a");
        let mut b = candidate_draft("c2", "family-b");
        a.summary = "Same human wording".into();
        b.summary = "Same human wording".into();
        let candidates = canonicalize_candidates(&intent, [a, b]).unwrap();
        assert_eq!(declared_family_count(&candidates), 2);
    }

    #[test]
    fn dispositions_are_candidate_only() {
        let intent = admit_intent(intent_draft()).unwrap();
        let candidate = admit_candidate(&intent, candidate_draft("c1", "family-a")).unwrap();
        let record = record_disposition(
            &candidate,
            CreativeDispositionV1::ElegantButUnproven,
            Some("Worth specifying, but not yet tested.".into()),
        )
        .unwrap();
        assert!(!record.selected_for_specification());
        assert_eq!(record.candidate_id(), "c1");
    }

    #[test]
    fn selected_for_specification_is_not_a_verification_result() {
        let intent = admit_intent(intent_draft()).unwrap();
        let candidate = admit_candidate(&intent, candidate_draft("c1", "family-a")).unwrap();
        let record = record_disposition(
            &candidate,
            CreativeDispositionV1::SelectedForSpecification,
            None,
        )
        .unwrap();
        assert!(record.selected_for_specification());
        assert_eq!(record.disposition(), CreativeDispositionV1::SelectedForSpecification);
    }

    #[test]
    fn conflicting_duplicate_candidate_ids_fail_closed() {
        let intent = admit_intent(intent_draft()).unwrap();
        let first = candidate_draft("same-id", "family-a");
        let mut second = first.clone();
        second.summary = "Conflicting content".into();
        assert_eq!(
            canonicalize_candidates(&intent, [first, second]),
            Err(StudioError::DuplicateCandidateConflict {
                candidate_id: "same-id".into()
            })
        );
    }

    #[test]
    fn exact_duplicate_candidates_normalize() {
        let intent = admit_intent(intent_draft()).unwrap();
        let draft = candidate_draft("same-id", "family-a");
        let candidates = canonicalize_candidates(&intent, [draft.clone(), draft]).unwrap();
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn candidate_permutation_does_not_change_declared_family_census() {
        let intent = admit_intent(intent_draft()).unwrap();
        let a = candidate_draft("a", "family-a");
        let b = candidate_draft("b", "family-b");
        let c = candidate_draft("c", "family-a");
        let one = canonicalize_candidates(&intent, [a.clone(), b.clone(), c.clone()]).unwrap();
        let two = canonicalize_candidates(&intent, [c, a, b]).unwrap();
        assert_eq!(declared_family_census(&one), declared_family_census(&two));
    }
}
