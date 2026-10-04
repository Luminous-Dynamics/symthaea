// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit lexical and morphosyntactic binding between semantic formulation and phonology.
//!
//! This module is deliberately a contract, not a language generator. Lexical choices,
//! grammatical features, dependencies, agreement, ordering rules, and morphophonological
//! forms must be supplied explicitly with provenance. Unsupported language rules remain
//! unbound rather than being guessed.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::linguistic_frame::LinguisticFrame;

pub const LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION: &str =
    "broca-lexical-morphosyntactic-binding-v1";

/// Whether a language-specific morphosyntactic rule has been explicitly supplied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LanguageRuleStatus {
    Bound,
    Unbound,
}

/// Explicit provenance for language-specific realization rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageRuleBinding {
    pub language_tag: String,
    pub status: LanguageRuleStatus,
    /// Stable identifier for an explicitly supported language rule set.
    pub rule_id: Option<String>,
    /// Provenance for the rule set when it is bound.
    pub provenance: Option<String>,
    /// Why the language rule surface is intentionally left unbound.
    pub unbound_reason: Option<String>,
}

/// Source of a lexical item.
///
/// Semantic constituents carry an exact role/prime lineage. Function words are explicitly
/// marked as inserted grammatical material and cannot masquerade as semantic payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LexicalSource {
    SemanticConstituent {
        role: String,
        prime: String,
    },
    InsertedFunctionWord {
        insertion_reason: String,
    },
}

/// Controlled grammatical-function vocabulary with an escape hatch for explicit language
/// specific functions that are not normalized by the core contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammaticalFunction {
    Subject,
    Verb,
    Object,
    Oblique,
    Predicate,
    Modifier,
    Determiner,
    Auxiliary,
    Complement,
    FunctionWord,
    Other(String),
}

/// A single explicit lexical choice attached to a formulated constituent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexemeBinding {
    /// Zero-based position in the final linearized constituent stream.
    pub position: usize,
    pub source: LexicalSource,
    /// Canonical dictionary form.
    pub lemma: String,
    /// Identifier for the selected lexical entry.
    pub lexeme_id: String,
    pub grammatical_function: GrammaticalFunction,
    /// Explicit inflectional/morphological features.
    pub morphology: Vec<MorphologicalFeature>,
    /// Morphophonological form after inflection but before phoneme realization.
    pub morphophonological_form: Option<String>,
    /// Source/provenance for the lexical decision.
    pub provenance: String,
    /// True only for semantic payload; false for inserted grammatical material.
    pub semantic_payload: bool,
}

/// Generic typed feature pair so language-specific morphology can be represented without
/// pretending the core contract supports a fixed inventory of languages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphologicalFeature {
    pub category: String,
    pub value: String,
}

/// A typed dependency between two linearized constituents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstituentDependency {
    pub governor_position: usize,
    pub dependent_position: usize,
    /// Language-specific dependency label may be used when the normalized set is insufficient.
    pub relation: String,
}

/// A typed agreement requirement. The controller and target must both explicitly carry the
/// requested feature/value; the contract never infers the target feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreementConstraint {
    pub controller_position: usize,
    pub target_position: usize,
    pub feature: MorphologicalFeature,
}

/// Complete lexical + morphosyntactic binding for one LinguisticFrame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalMorphosyntacticBinding {
    pub version: String,
    /// Exact upstream linguistic contract version.
    pub source_frame_version: String,
    /// Exact upstream grounding surface, used as an anti-stale lineage check.
    pub source_frame_grounding: String,
    pub language: LanguageRuleBinding,
    pub constituents: Vec<LexemeBinding>,
    pub dependencies: Vec<ConstituentDependency>,
    pub agreement: Vec<AgreementConstraint>,
}

impl LexicalMorphosyntacticBinding {
    /// Construct a binding from explicit caller-supplied lexical and grammatical data.
    ///
    /// No lexical item, function word, morphology, dependency, agreement relation, or
    /// language rule is synthesized by this constructor.
    pub fn new(
        frame: &LinguisticFrame,
        language: LanguageRuleBinding,
        constituents: Vec<LexemeBinding>,
        dependencies: Vec<ConstituentDependency>,
        agreement: Vec<AgreementConstraint>,
    ) -> Result<Self, LexicalBindingError> {
        frame.validate()
            .map_err(|_| LexicalBindingError::UpstreamMismatch)?;

        let binding = Self {
            version: LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION.to_string(),
            source_frame_version: frame.version.clone(),
            source_frame_grounding: frame.grounding_surface(),
            language,
            constituents,
            dependencies,
            agreement,
        };
        binding.validate_against_frame(frame)?;
        Ok(binding)
    }

    /// Validate persisted/deserialized cross-field invariants and exact upstream lineage.
    pub fn validate_against_frame(
        &self,
        frame: &LinguisticFrame,
    ) -> Result<(), LexicalBindingError> {
        frame.validate()
            .map_err(|_| LexicalBindingError::UpstreamMismatch)?;
        self.validate()?;

        if self.source_frame_version != frame.version
            || self.source_frame_grounding != frame.grounding_surface()
        {
            return Err(LexicalBindingError::UpstreamMismatch);
        }

        let expected_sources: HashSet<(String, String)> = frame
            .constituents
            .iter()
            .map(|slot| (slot.role.clone(), slot.prime.clone()))
            .collect();

        let mut observed_sources = HashSet::new();
        for constituent in &self.constituents {
            match &constituent.source {
                LexicalSource::SemanticConstituent { role, prime } => {
                    if !expected_sources.contains(&(role.clone(), prime.clone())) {
                        return Err(LexicalBindingError::UnknownSemanticSource {
                            role: role.clone(),
                            prime: prime.clone(),
                        });
                    }
                    if !observed_sources.insert((role.clone(), prime.clone())) {
                        return Err(LexicalBindingError::DuplicateSemanticSource {
                            role: role.clone(),
                            prime: prime.clone(),
                        });
                    }
                }
                LexicalSource::InsertedFunctionWord { .. } => {}
            }
        }

        if observed_sources.len() != expected_sources.len() {
            let missing = expected_sources
                .difference(&observed_sources)
                .next()
                .cloned()
                .ok_or(LexicalBindingError::MissingSemanticCoverage {
                    role: "unknown".to_string(),
                    prime: "unknown".to_string(),
                })?;
            return Err(LexicalBindingError::MissingSemanticCoverage {
                role: missing.0,
                prime: missing.1,
            });
        }

        Ok(())
    }

    /// Validate this contract without requiring an upstream frame object.
    pub fn validate(&self) -> Result<(), LexicalBindingError> {
        if self.version != LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION {
            return Err(LexicalBindingError::InvalidVersion);
        }
        if self.source_frame_version.trim().is_empty()
            || self.source_frame_grounding.trim().is_empty()
        {
            return Err(LexicalBindingError::MissingUpstreamLineage);
        }

        validate_language_binding(&self.language)?;
        validate_constituents(&self.constituents)?;
        validate_dependencies(&self.constituents, &self.dependencies)?;
        validate_agreement(&self.constituents, &self.agreement)?;

        Ok(())
    }

    /// Stable evidence surface suitable for append-only measurement artifacts.
    pub fn grounding_surface(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"version\":\"{}\",\"serialization\":\"failed\"}}",
                self.version
            )
        })
    }

    /// Stable provenance token that can be carried into a downstream phonological plan.
    pub fn provenance_token(&self) -> String {
        let digest = blake3::hash(self.grounding_surface().as_bytes());
        digest.to_hex().to_string()
    }
}

fn validate_language_binding(binding: &LanguageRuleBinding) -> Result<(), LexicalBindingError> {
    if binding.language_tag.trim().is_empty() {
        return Err(LexicalBindingError::EmptyLanguageTag);
    }

    match binding.status {
        LanguageRuleStatus::Bound => {
            let rule_id = binding
                .rule_id
                .as_deref()
                .ok_or(LexicalBindingError::BoundRuleMissingId)?;
            let provenance = binding
                .provenance
                .as_deref()
                .ok_or(LexicalBindingError::BoundRuleMissingProvenance)?;

            if rule_id.trim().is_empty() {
                return Err(LexicalBindingError::BoundRuleMissingId);
            }
            if provenance.trim().is_empty() {
                return Err(LexicalBindingError::BoundRuleMissingProvenance);
            }
            if binding.unbound_reason.is_some() {
                return Err(LexicalBindingError::BoundRuleHasUnboundReason);
            }
        }
        LanguageRuleStatus::Unbound => {
            if binding.rule_id.is_some() || binding.provenance.is_some() {
                return Err(LexicalBindingError::UnboundRuleCarriesBinding);
            }
            let reason = binding
                .unbound_reason
                .as_deref()
                .ok_or(LexicalBindingError::UnboundRuleMissingReason)?;
            if reason.trim().is_empty() {
                return Err(LexicalBindingError::UnboundRuleMissingReason);
            }
        }
    }

    Ok(())
}

fn validate_constituents(constituents: &[LexemeBinding]) -> Result<(), LexicalBindingError> {
    let mut semantic_count = 0usize;

    for (expected_position, constituent) in constituents.iter().enumerate() {
        if constituent.position != expected_position {
            return Err(LexicalBindingError::NonContiguousPositions);
        }
        if constituent.lemma.trim().is_empty() {
            return Err(LexicalBindingError::EmptyLemma {
                position: constituent.position,
            });
        }
        if constituent.lexeme_id.trim().is_empty() {
            return Err(LexicalBindingError::EmptyLexemeId {
                position: constituent.position,
            });
        }
        if constituent.provenance.trim().is_empty() {
            return Err(LexicalBindingError::EmptyLexicalProvenance {
                position: constituent.position,
            });
        }

        match &constituent.source {
            LexicalSource::SemanticConstituent { role, prime } => {
                semantic_count += 1;
                if role.trim().is_empty() || prime.trim().is_empty() {
                    return Err(LexicalBindingError::EmptySemanticSource {
                        position: constituent.position,
                    });
                }
                if !constituent.semantic_payload {
                    return Err(LexicalBindingError::SemanticPayloadMismatch {
                        position: constituent.position,
                    });
                }
                if matches!(constituent.grammatical_function, GrammaticalFunction::FunctionWord) {
                    return Err(LexicalBindingError::SemanticFunctionMismatch {
                        position: constituent.position,
                    });
                }
            }
            LexicalSource::InsertedFunctionWord { insertion_reason } => {
                if insertion_reason.trim().is_empty() {
                    return Err(LexicalBindingError::EmptyFunctionWordReason {
                        position: constituent.position,
                    });
                }
                if constituent.semantic_payload {
                    return Err(LexicalBindingError::SemanticPayloadMismatch {
                        position: constituent.position,
                    });
                }
                if !matches!(
                    constituent.grammatical_function,
                    GrammaticalFunction::FunctionWord
                ) {
                    return Err(LexicalBindingError::FunctionWordFunctionMismatch {
                        position: constituent.position,
                    });
                }
            }
        }

        let mut categories = HashSet::new();
        for feature in &constituent.morphology {
            if feature.category.trim().is_empty() || feature.value.trim().is_empty() {
                return Err(LexicalBindingError::EmptyMorphologicalFeature {
                    position: constituent.position,
                });
            }
            if !categories.insert(feature.category.clone()) {
                return Err(LexicalBindingError::ConflictingMorphology {
                    position: constituent.position,
                    category: feature.category.clone(),
                });
            }
        }

        if let Some(form) = constituent.morphophonological_form.as_deref() {
            if form.trim().is_empty() {
                return Err(LexicalBindingError::EmptyMorphophonologicalForm {
                    position: constituent.position,
                });
            }
        }
    }

    if semantic_count == 0 {
        return Err(LexicalBindingError::NoSemanticPayload);
    }

    Ok(())
}

fn validate_dependencies(
    constituents: &[LexemeBinding],
    dependencies: &[ConstituentDependency],
) -> Result<(), LexicalBindingError> {
    let mut seen = HashSet::new();
    for dependency in dependencies {
        if dependency.governor_position >= constituents.len()
            || dependency.dependent_position >= constituents.len()
        {
            return Err(LexicalBindingError::DependencyPositionOutOfRange);
        }
        if dependency.governor_position == dependency.dependent_position {
            return Err(LexicalBindingError::SelfDependency);
        }
        if dependency.relation.trim().is_empty() {
            return Err(LexicalBindingError::EmptyDependencyRelation);
        }

        if !seen.insert((
            dependency.governor_position,
            dependency.dependent_position,
            dependency.relation.clone(),
        )) {
            return Err(LexicalBindingError::DuplicateDependency);
        }
    }
    Ok(())
}

fn validate_agreement(
    constituents: &[LexemeBinding],
    agreement: &[AgreementConstraint],
) -> Result<(), LexicalBindingError> {
    let mut seen = HashSet::new();

    for constraint in agreement {
        if constraint.controller_position >= constituents.len()
            || constraint.target_position >= constituents.len()
        {
            return Err(LexicalBindingError::AgreementPositionOutOfRange);
        }
        if constraint.controller_position == constraint.target_position {
            return Err(LexicalBindingError::SelfAgreement);
        }
        if constraint.feature.category.trim().is_empty()
            || constraint.feature.value.trim().is_empty()
        {
            return Err(LexicalBindingError::EmptyAgreementFeature);
        }

        let key = (
            constraint.controller_position,
            constraint.target_position,
            constraint.feature.clone(),
        );
        if !seen.insert(key) {
            return Err(LexicalBindingError::DuplicateAgreement);
        }

        let controller_has = constituents[constraint.controller_position]
            .morphology
            .iter()
            .any(|feature| feature == &constraint.feature);
        let target_has = constituents[constraint.target_position]
            .morphology
            .iter()
            .any(|feature| feature == &constraint.feature);

        if !controller_has || !target_has {
            return Err(LexicalBindingError::AgreementFeatureMismatch);
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalBindingError {
    InvalidVersion,
    MissingUpstreamLineage,
    UpstreamMismatch,
    EmptyLanguageTag,
    BoundRuleMissingId,
    BoundRuleMissingProvenance,
    BoundRuleHasUnboundReason,
    UnboundRuleCarriesBinding,
    UnboundRuleMissingReason,
    NonContiguousPositions,
    EmptyLemma { position: usize },
    EmptyLexemeId { position: usize },
    EmptyLexicalProvenance { position: usize },
    EmptySemanticSource { position: usize },
    UnknownSemanticSource { role: String, prime: String },
    DuplicateSemanticSource { role: String, prime: String },
    MissingSemanticCoverage { role: String, prime: String },
    NoSemanticPayload,
    SemanticPayloadMismatch { position: usize },
    SemanticFunctionMismatch { position: usize },
    EmptyFunctionWordReason { position: usize },
    FunctionWordFunctionMismatch { position: usize },
    EmptyMorphologicalFeature { position: usize },
    ConflictingMorphology { position: usize, category: String },
    EmptyMorphophonologicalForm { position: usize },
    DependencyPositionOutOfRange,
    SelfDependency,
    EmptyDependencyRelation,
    DuplicateDependency,
    AgreementPositionOutOfRange,
    SelfAgreement,
    EmptyAgreementFeature,
    DuplicateAgreement,
    AgreementFeatureMismatch,
}

impl std::fmt::Display for LexicalBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "lexical/morphosyntactic binding version is unsupported"),
            Self::MissingUpstreamLineage => write!(f, "upstream linguistic-frame lineage is missing"),
            Self::UpstreamMismatch => write!(f, "lexical binding no longer matches its upstream linguistic frame"),
            Self::EmptyLanguageTag => write!(f, "language tag must be non-empty"),
            Self::BoundRuleMissingId => write!(f, "bound language rules require a non-empty rule id"),
            Self::BoundRuleMissingProvenance => write!(f, "bound language rules require provenance"),
            Self::BoundRuleHasUnboundReason => write!(f, "bound language rules must not carry an unbound reason"),
            Self::UnboundRuleCarriesBinding => write!(f, "unbound language rules must not carry a rule id or provenance"),
            Self::UnboundRuleMissingReason => write!(f, "unbound language rules require an explicit reason"),
            Self::NonContiguousPositions => write!(f, "lexical positions must be contiguous from zero"),
            Self::EmptyLemma { position } => write!(f, "lexical position {position} has an empty lemma"),
            Self::EmptyLexemeId { position } => write!(f, "lexical position {position} has an empty lexeme id"),
            Self::EmptyLexicalProvenance { position } => write!(f, "lexical position {position} has empty provenance"),
            Self::EmptySemanticSource { position } => write!(f, "semantic lexical source at position {position} is incomplete"),
            Self::UnknownSemanticSource { role, prime } => write!(f, "lexical binding references unknown semantic source {role}:{prime}"),
            Self::DuplicateSemanticSource { role, prime } => write!(f, "semantic source {role}:{prime} is bound more than once"),
            Self::MissingSemanticCoverage { role, prime } => write!(f, "semantic source {role}:{prime} is not lexically covered"),
            Self::NoSemanticPayload => write!(f, "lexical binding must contain semantic payload"),
            Self::SemanticPayloadMismatch { position } => write!(f, "semantic-payload flag disagrees with lexical source at position {position}"),
            Self::SemanticFunctionMismatch { position } => write!(f, "semantic payload at position {position} cannot be classified as a function word"),
            Self::EmptyFunctionWordReason { position } => write!(f, "inserted function word at position {position} requires an insertion reason"),
            Self::FunctionWordFunctionMismatch { position } => write!(f, "inserted function word at position {position} must use the FunctionWord grammatical function"),
            Self::EmptyMorphologicalFeature { position } => write!(f, "lexical position {position} contains an empty morphological feature"),
            Self::ConflictingMorphology { position, category } => write!(f, "lexical position {position} contains conflicting morphology for category {category}"),
            Self::EmptyMorphophonologicalForm { position } => write!(f, "lexical position {position} has an empty morphophonological form"),
            Self::DependencyPositionOutOfRange => write!(f, "dependency references an out-of-range constituent position"),
            Self::SelfDependency => write!(f, "dependency cannot point from a constituent to itself"),
            Self::EmptyDependencyRelation => write!(f, "dependency relation must be non-empty"),
            Self::DuplicateDependency => write!(f, "duplicate dependency is not permitted"),
            Self::AgreementPositionOutOfRange => write!(f, "agreement references an out-of-range constituent position"),
            Self::SelfAgreement => write!(f, "agreement cannot target the same constituent"),
            Self::EmptyAgreementFeature => write!(f, "agreement feature must be non-empty"),
            Self::DuplicateAgreement => write!(f, "duplicate agreement constraint is not permitted"),
            Self::AgreementFeatureMismatch => write!(f, "agreement feature must be explicitly present on both controller and target"),
        }
    }
}

impl std::error::Error for LexicalBindingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic_frame::{FormulationStrategy, LinguisticBindingStatus};
    use crate::{StructuredDecoder, ThoughtChannels};
    use symthaea_core::genesis::GenesisSeed;

    fn statement_frame() -> LinguisticFrame {
        let genesis = GenesisSeed::from_phrase("lexical-binding-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        LinguisticFrame::from_speech_plan(&crate::SpeechPlan::from_readout(&channels, &readout))
    }

    fn base_semantic_bindings() -> Vec<LexemeBinding> {
        vec![
            LexemeBinding {
                position: 0,
                source: LexicalSource::SemanticConstituent {
                    role: "AGENT".into(),
                    prime: "I".into(),
                },
                lemma: "I".into(),
                lexeme_id: "en:pronoun:I".into(),
                grammatical_function: GrammaticalFunction::Subject,
                morphology: vec![
                    MorphologicalFeature {
                        category: "person".into(),
                        value: "1".into(),
                    },
                    MorphologicalFeature {
                        category: "number".into(),
                        value: "singular".into(),
                    },
                ],
                morphophonological_form: Some("I".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: true,
            },
            LexemeBinding {
                position: 1,
                source: LexicalSource::SemanticConstituent {
                    role: "ACTION".into(),
                    prime: "DO".into(),
                },
                lemma: "do".into(),
                lexeme_id: "en:verb:do".into(),
                grammatical_function: GrammaticalFunction::Verb,
                morphology: vec![
                    MorphologicalFeature {
                        category: "person".into(),
                        value: "1".into(),
                    },
                    MorphologicalFeature {
                        category: "number".into(),
                        value: "singular".into(),
                    },
                ],
                morphophonological_form: Some("do".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: true,
            },
            LexemeBinding {
                position: 2,
                source: LexicalSource::SemanticConstituent {
                    role: "PATIENT".into(),
                    prime: "SOMETHING".into(),
                },
                lemma: "something".into(),
                lexeme_id: "en:pronoun:something".into(),
                grammatical_function: GrammaticalFunction::Object,
                morphology: vec![],
                morphophonological_form: Some("something".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: true,
            },
            LexemeBinding {
                position: 3,
                source: LexicalSource::SemanticConstituent {
                    role: "PREDICATE".into(),
                    prime: "KNOW".into(),
                },
                lemma: "know".into(),
                lexeme_id: "en:verb:know".into(),
                grammatical_function: GrammaticalFunction::Predicate,
                morphology: vec![],
                morphophonological_form: Some("know".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: true,
            },
            LexemeBinding {
                position: 4,
                source: LexicalSource::SemanticConstituent {
                    role: "EVALUATOR".into(),
                    prime: "TRUE".into(),
                },
                lemma: "true".into(),
                lexeme_id: "en:adj:true".into(),
                grammatical_function: GrammaticalFunction::Predicate,
                morphology: vec![],
                morphophonological_form: Some("true".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: true,
            },
        ]
    }

    fn english_rule_binding() -> LanguageRuleBinding {
        LanguageRuleBinding {
            language_tag: "en".into(),
            status: LanguageRuleStatus::Bound,
            rule_id: Some("toy-en-svo-v1".into()),
            provenance: Some("fixture:english-rules-v1".into()),
            unbound_reason: None,
        }
    }

    #[test]
    fn explicit_semantic_coverage_and_lineage_pass() {
        let frame = statement_frame();
        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            base_semantic_bindings(),
            vec![
                ConstituentDependency {
                    governor_position: 1,
                    dependent_position: 0,
                    relation: "subject".into(),
                },
                ConstituentDependency {
                    governor_position: 1,
                    dependent_position: 2,
                    relation: "object".into(),
                },
            ],
            vec![AgreementConstraint {
                controller_position: 0,
                target_position: 1,
                feature: MorphologicalFeature {
                    category: "person".into(),
                    value: "1".into(),
                },
            }],
        )
        .expect("explicit fixture should bind");
        assert!(binding.validate_against_frame(&frame).is_ok());
        assert!(binding.grounding_surface().contains(LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION));
        assert_eq!(binding.provenance_token().len(), 64);
    }

    #[test]
    fn function_word_is_not_semantic_payload() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings.insert(
            1,
            LexemeBinding {
                position: 1,
                source: LexicalSource::InsertedFunctionWord {
                    insertion_reason: "fixture:explicit-determiner".into(),
                },
                lemma: "the".into(),
                lexeme_id: "en:det:the".into(),
                grammatical_function: GrammaticalFunction::FunctionWord,
                morphology: vec![],
                morphophonological_form: Some("the".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: false,
            },
        );
        for (position, binding) in bindings.iter_mut().enumerate() {
            binding.position = position;
        }

        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect("function-word insertion must remain explicit");
        assert_eq!(
            binding
                .constituents
                .iter()
                .filter(|item| !item.semantic_payload)
                .count(),
            1
        );
    }

    #[test]
    fn unsupported_language_rule_remains_unbound() {
        let frame = statement_frame();
        let language = LanguageRuleBinding {
            language_tag: "xx".into(),
            status: LanguageRuleStatus::Unbound,
            rule_id: None,
            provenance: None,
            unbound_reason: Some("no-supported-rule-set".into()),
        };
        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            language,
            base_semantic_bindings(),
            vec![],
            vec![],
        )
        .expect("unsupported language may remain explicitly unbound");
        assert_eq!(binding.language.status, LanguageRuleStatus::Unbound);
    }

    #[test]
    fn missing_semantic_source_fails_closed() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings.pop();
        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect_err("missing source must fail closed");
        assert!(matches!(error, LexicalBindingError::MissingSemanticCoverage { .. }));
    }

    #[test]
    fn agreement_does_not_infer_missing_feature() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings[1].morphology.clear();
        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![AgreementConstraint {
                controller_position: 0,
                target_position: 1,
                feature: MorphologicalFeature {
                    category: "person".into(),
                    value: "1".into(),
                },
            }],
        )
        .expect_err("agreement must not manufacture the feature");
        assert_eq!(error, LexicalBindingError::AgreementFeatureMismatch);
    }

    #[test]
    fn conflicting_morphology_fails_closed() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings[0].morphology.push(MorphologicalFeature {
            category: "number".into(),
            value: "plural".into(),
        });
        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect_err("duplicate morphology category must fail");
        assert_eq!(
            error,
            LexicalBindingError::ConflictingMorphology {
                position: 0,
                category: "number".into()
            }
        );
    }

    #[test]
    fn stale_frame_grounding_fails_closed() {
        let frame = statement_frame();
        let mut binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            base_semantic_bindings(),
            vec![],
            vec![],
        )
        .unwrap();
        binding.source_frame_grounding.push_str(";tampered");
        assert_eq!(
            binding.validate_against_frame(&frame)
                .expect_err("stale lineage must fail"),
            LexicalBindingError::UpstreamMismatch
        );
    }

    #[test]
    fn semantic_function_word_mismatch_fails_closed() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings[0].grammatical_function = GrammaticalFunction::FunctionWord;
        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect_err("semantic item must not masquerade as function word");
        assert_eq!(
            error,
            LexicalBindingError::SemanticFunctionMismatch { position: 0 }
        );
    }

    #[test]
    fn insertion_reason_is_required() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings.insert(
            1,
            LexemeBinding {
                position: 1,
                source: LexicalSource::InsertedFunctionWord {
                    insertion_reason: " ".into(),
                },
                lemma: "to".into(),
                lexeme_id: "en:function:to".into(),
                grammatical_function: GrammaticalFunction::FunctionWord,
                morphology: vec![],
                morphophonological_form: Some("to".into()),
                provenance: "fixture:en:v1".into(),
                semantic_payload: false,
            },
        );
        for (position, item) in bindings.iter_mut().enumerate() {
            item.position = position;
        }

        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect_err("empty insertion reason must fail");
        assert_eq!(
            error,
            LexicalBindingError::EmptyFunctionWordReason { position: 1 }
        );
    }

    #[test]
    fn module_uses_existing_linguistic_binding_status_surface() {
        assert_eq!(
            LinguisticBindingStatus::RoleStructureOnly,
            LinguisticBindingStatus::RoleStructureOnly
        );
        assert_eq!(FormulationStrategy::Declarative as u8, 0);
    }
}
