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

use crate::linguistic_frame::{FormulationStrategy, LinguisticFrame};

pub const LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION: &str =
    "broca-lexical-morphosyntactic-binding-v2";

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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Stable identity for explicit morphophonological derivation traces.
pub const MORPHOPHONOLOGICAL_DERIVATION_WITNESS_VERSION: &str =
    "broca-morphophonological-derivation-witness-v1";

pub const MORPHOPHONOLOGICAL_RULE_SET_VERSION: &str =
    "broca-morphophonological-rule-set-v1";
pub const MORPHOPHONOLOGICAL_RULE_SELECTION_POLICY: &str =
    "exact-feature-single-rule-v1";
pub const MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION: &str =
    "broca-morphophonological-resource-evidence-v1";

/// Whether an executable morphology resource is externally sourced or explicitly
/// authored as a local/fixture resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MorphophonologicalResourceOrigin {
    External,
    HandAuthored,
}

/// Provenance for the linguistic/resource artifact behind an executable rule set.
///
/// External resources require a source URI, immutable revision identifier, and declared
/// license. Hand-authored resources intentionally carry no fabricated external attribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalResourceEvidence {
    pub version: String,
    pub origin: MorphophonologicalResourceOrigin,
    pub source_id: String,
    pub source_uri: Option<String>,
    pub revision: String,
    pub license: Option<String>,
}

impl MorphophonologicalResourceEvidence {
    pub fn hand_authored(
        source_id: impl Into<String>,
        revision: impl Into<String>,
    ) -> Result<Self, MorphophonologicalResourceEvidenceError> {
        let evidence = Self {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.to_string(),
            origin: MorphophonologicalResourceOrigin::HandAuthored,
            source_id: source_id.into(),
            source_uri: None,
            revision: revision.into(),
            license: None,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn external(
        source_id: impl Into<String>,
        source_uri: impl Into<String>,
        revision: impl Into<String>,
        license: impl Into<String>,
    ) -> Result<Self, MorphophonologicalResourceEvidenceError> {
        let evidence = Self {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.to_string(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: source_id.into(),
            source_uri: Some(source_uri.into()),
            revision: revision.into(),
            license: Some(license.into()),
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), MorphophonologicalResourceEvidenceError> {
        if self.version != MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION {
            return Err(MorphophonologicalResourceEvidenceError::InvalidVersion);
        }
        if self.source_id.trim().is_empty() {
            return Err(MorphophonologicalResourceEvidenceError::EmptySourceId);
        }
        if self.revision.trim().is_empty() {
            return Err(MorphophonologicalResourceEvidenceError::EmptyRevision);
        }

        match self.origin {
            MorphophonologicalResourceOrigin::External => {
                let uri = self
                    .source_uri
                    .as_deref()
                    .ok_or(MorphophonologicalResourceEvidenceError::ExternalMissingUri)?;
                let license = self
                    .license
                    .as_deref()
                    .ok_or(MorphophonologicalResourceEvidenceError::ExternalMissingLicense)?;
                if uri.trim().is_empty() {
                    return Err(MorphophonologicalResourceEvidenceError::ExternalMissingUri);
                }
                if license.trim().is_empty() {
                    return Err(MorphophonologicalResourceEvidenceError::ExternalMissingLicense);
                }
            }
            MorphophonologicalResourceOrigin::HandAuthored => {
                if self.source_uri.is_some() || self.license.is_some() {
                    return Err(
                        MorphophonologicalResourceEvidenceError::HandAuthoredCarriesExternalMetadata,
                    );
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphophonologicalResourceEvidenceError {
    InvalidVersion,
    EmptySourceId,
    EmptyRevision,
    ExternalMissingUri,
    ExternalMissingLicense,
    HandAuthoredCarriesExternalMetadata,
}

impl std::fmt::Display for MorphophonologicalResourceEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "morphophonological resource-evidence version is unsupported"),
            Self::EmptySourceId => write!(f, "morphophonological resource source id must be non-empty"),
            Self::EmptyRevision => write!(f, "morphophonological resource revision must be non-empty"),
            Self::ExternalMissingUri => write!(f, "external morphophonological resources require a source URI"),
            Self::ExternalMissingLicense => write!(f, "external morphophonological resources require a declared license"),
            Self::HandAuthoredCarriesExternalMetadata => write!(f, "hand-authored morphophonological resources must not carry fabricated external URI or license metadata"),
        }
    }
}

impl std::error::Error for MorphophonologicalResourceEvidenceError {}

/// A deliberately small deterministic operation vocabulary for executable
/// morphophonological derivation evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MorphophonologicalRuleOperation {
    Identity,
    AppendSuffix { suffix: String },
    PrependPrefix { prefix: String },
    ReplaceSuffix { from: String, to: String },
    ReplacePrefix { from: String, to: String },
    ReplaceExact { from: String, to: String },
}

/// One executable rule selected by an exact morphological feature set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalRule {
    pub rule_id: String,
    pub morphology: Vec<MorphologicalFeature>,
    pub operation: MorphophonologicalRuleOperation,
}

/// Explicit executable rule surface used to replay morphophonological derivations.
///
/// The rule set is intentionally narrower than a general finite-state grammar. It provides
/// deterministic replayable evidence for a small, inspectable rule vocabulary; extending the
/// vocabulary is a separate capability/qualification boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalRuleSet {
    pub version: String,
    pub language_tag: String,
    /// Provenance identifier for the linguistic resource or hand-authored resource snapshot.
    pub source_id: String,
    /// Explicit dialect/scope claim; this is not inferred from the language tag.
    pub dialect_scope: String,
    /// Exact rule-selection policy used by the executable engine.
    pub selection_policy: String,
    /// Typed origin/provenance for the executable resource behind this rule set.
    pub resource_evidence: MorphophonologicalResourceEvidence,
    pub rule_id: String,
    pub provenance: String,
    pub rules: Vec<MorphophonologicalRule>,
}

impl MorphophonologicalRuleSet {
    pub fn new(
        language_tag: impl Into<String>,
        source_id: impl Into<String>,
        dialect_scope: impl Into<String>,
        resource_evidence: MorphophonologicalResourceEvidence,
        rule_id: impl Into<String>,
        provenance: impl Into<String>,
        rules: Vec<MorphophonologicalRule>,
    ) -> Result<Self, MorphophonologicalRuleSetError> {
        let rule_set = Self {
            version: MORPHOPHONOLOGICAL_RULE_SET_VERSION.to_string(),
            language_tag: language_tag.into(),
            source_id: source_id.into(),
            dialect_scope: dialect_scope.into(),
            selection_policy: MORPHOPHONOLOGICAL_RULE_SELECTION_POLICY.to_string(),
            resource_evidence,
            rule_id: rule_id.into(),
            provenance: provenance.into(),
            rules,
        };
        rule_set.validate()?;
        Ok(rule_set)
    }

    pub fn validate(&self) -> Result<(), MorphophonologicalRuleSetError> {
        if self.version != MORPHOPHONOLOGICAL_RULE_SET_VERSION {
            return Err(MorphophonologicalRuleSetError::InvalidVersion);
        }
        if self.language_tag.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyLanguageTag);
        }
        if self.source_id.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptySourceId);
        }
        if self.dialect_scope.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyDialectScope);
        }
        if self.selection_policy != MORPHOPHONOLOGICAL_RULE_SELECTION_POLICY {
            return Err(MorphophonologicalRuleSetError::UnsupportedSelectionPolicy);
        }
        self.resource_evidence
            .validate()
            .map_err(|_| MorphophonologicalRuleSetError::InvalidResourceEvidence)?;
        if self.source_id != self.resource_evidence.source_id {
            return Err(MorphophonologicalRuleSetError::ResourceSourceIdMismatch);
        }
        if self.rule_id.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyRuleSetId);
        }
        if self.provenance.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyProvenance);
        }
        if self.rules.is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyRuleSet);
        }

        let mut rule_ids = HashSet::new();
        let mut feature_signatures = HashSet::new();
        for rule in &self.rules {
            if rule.rule_id.trim().is_empty() {
                return Err(MorphophonologicalRuleSetError::EmptyRuleId);
            }
            if !rule_ids.insert(rule.rule_id.clone()) {
                return Err(MorphophonologicalRuleSetError::DuplicateRuleId);
            }

            let mut categories = HashSet::new();
            for feature in &rule.morphology {
                if feature.category.trim().is_empty() || feature.value.trim().is_empty() {
                    return Err(MorphophonologicalRuleSetError::InvalidMorphology);
                }
                if !categories.insert(feature.category.clone()) {
                    return Err(MorphophonologicalRuleSetError::InvalidMorphology);
                }
            }

            let canonical = canonical_morphology(&rule.morphology);
            if !feature_signatures.insert(canonical) {
                return Err(MorphophonologicalRuleSetError::AmbiguousFeatureMatch);
            }

            validate_rule_operation(&rule.operation)?;
        }

        Ok(())
    }

    /// Deterministically derive a surface form from a lemma and exact morphology.
    pub fn derive(
        &self,
        lemma: &str,
        morphology: &[MorphologicalFeature],
    ) -> Result<(String, String), MorphophonologicalRuleSetError> {
        self.validate()?;
        if lemma.trim().is_empty() {
            return Err(MorphophonologicalRuleSetError::EmptyLemma);
        }

        let expected = canonical_morphology(morphology);
        if expected.len() != morphology.len() {
            return Err(MorphophonologicalRuleSetError::InvalidMorphology);
        }

        let matches = self
            .rules
            .iter()
            .filter(|rule| canonical_morphology(&rule.morphology) == expected)
            .collect::<Vec<_>>();

        let rule = match matches.as_slice() {
            [] => return Err(MorphophonologicalRuleSetError::NoMatchingRule),
            [rule] => rule,
            _ => return Err(MorphophonologicalRuleSetError::AmbiguousFeatureMatch),
        };

        let output = apply_rule_operation(lemma, &rule.operation)?;
        Ok((output, rule.rule_id.clone()))
    }

    pub fn grounding_surface(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"version\":\"{}\",\"serialization\":\"failed\"}}",
                self.version
            )
        })
    }

    /// Domain-separated identity of the exact executable rule-set content.
    pub fn resource_blake3(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-morphophonological-rule-set-v1\\0");
        hasher.update(self.grounding_surface().as_bytes());
        hasher.update(b"symthaea-morphophonological-resource-evidence-v1\\0");
        hasher.update(self.resource_evidence.version.as_bytes());
        hasher.update(&(serde_json::to_string(&self.resource_evidence)
            .unwrap_or_else(|_| String::from("serialization-failed")))
            .len() as u64)
            .to_le_bytes());
        hasher.update(
            serde_json::to_string(&self.resource_evidence)
                .unwrap_or_else(|_| String::from("serialization-failed"))
                .as_bytes(),
        );
        hasher.finalize().to_hex().to_string()
    }

    pub fn provenance_token(&self) -> String {
        self.resource_blake3()
    }
}

fn canonical_morphology(morphology: &[MorphologicalFeature]) -> Vec<(String, String)> {
    let mut canonical = morphology
        .iter()
        .map(|feature| (feature.category.clone(), feature.value.clone()))
        .collect::<Vec<_>>();
    canonical.sort();
    canonical
}

fn validate_rule_operation(
    operation: &MorphophonologicalRuleOperation,
) -> Result<(), MorphophonologicalRuleSetError> {
    match operation {
        MorphophonologicalRuleOperation::Identity => {}
        MorphophonologicalRuleOperation::AppendSuffix { suffix }
        | MorphophonologicalRuleOperation::PrependPrefix { prefix: suffix } => {
            if suffix.is_empty() {
                return Err(MorphophonologicalRuleSetError::EmptyOperationOperand);
            }
        }
        MorphophonologicalRuleOperation::ReplaceSuffix { from, to }
        | MorphophonologicalRuleOperation::ReplacePrefix { from, to }
        | MorphophonologicalRuleOperation::ReplaceExact { from, to } => {
            if from.is_empty() || to.is_empty() {
                return Err(MorphophonologicalRuleSetError::EmptyOperationOperand);
            }
        }
    }
    Ok(())
}

fn apply_rule_operation(
    lemma: &str,
    operation: &MorphophonologicalRuleOperation,
) -> Result<String, MorphophonologicalRuleSetError> {
    let output = match operation {
        MorphophonologicalRuleOperation::Identity => lemma.to_string(),
        MorphophonologicalRuleOperation::AppendSuffix { suffix } => {
            format!("{lemma}{suffix}")
        }
        MorphophonologicalRuleOperation::PrependPrefix { prefix } => {
            format!("{prefix}{lemma}")
        }
        MorphophonologicalRuleOperation::ReplaceSuffix { from, to } => {
            let stem = lemma
                .strip_suffix(from)
                .ok_or(MorphophonologicalRuleSetError::SourceFormMismatch)?;
            format!("{stem}{to}")
        }
        MorphophonologicalRuleOperation::ReplacePrefix { from, to } => {
            let stem = lemma
                .strip_prefix(from)
                .ok_or(MorphophonologicalRuleSetError::SourceFormMismatch)?;
            format!("{to}{stem}")
        }
        MorphophonologicalRuleOperation::ReplaceExact { from, to } => {
            if lemma != from {
                return Err(MorphophonologicalRuleSetError::SourceFormMismatch);
            }
            to.clone()
        }
    };

    if output.trim().is_empty() {
        return Err(MorphophonologicalRuleSetError::EmptyDerivedForm);
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphophonologicalRuleSetError {
    InvalidVersion,
    EmptyLanguageTag,
    EmptySourceId,
    EmptyDialectScope,
    UnsupportedSelectionPolicy,
    InvalidResourceEvidence,
    ResourceSourceIdMismatch,
    EmptyRuleSetId,
    EmptyProvenance,
    EmptyRuleSet,
    EmptyRuleId,
    DuplicateRuleId,
    InvalidMorphology,
    AmbiguousFeatureMatch,
    EmptyOperationOperand,
    EmptyLemma,
    NoMatchingRule,
    SourceFormMismatch,
    EmptyDerivedForm,
}

impl std::fmt::Display for MorphophonologicalRuleSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "morphophonological rule-set version is unsupported"),
            Self::EmptyLanguageTag => write!(f, "morphophonological rule-set language tag must be non-empty"),
            Self::EmptySourceId => write!(f, "morphophonological rule-set source id must be non-empty"),
            Self::EmptyDialectScope => write!(f, "morphophonological rule-set dialect scope must be non-empty"),
            Self::UnsupportedSelectionPolicy => write!(f, "morphophonological rule-set selection policy is unsupported"),
            Self::InvalidResourceEvidence => write!(f, "morphophonological rule-set resource evidence is invalid"),
            Self::ResourceSourceIdMismatch => write!(f, "morphophonological rule-set source id must match its resource evidence source id"),
            Self::EmptyRuleSetId => write!(f, "morphophonological rule-set id must be non-empty"),
            Self::EmptyProvenance => write!(f, "morphophonological rule-set provenance must be non-empty"),
            Self::EmptyRuleSet => write!(f, "morphophonological rule set must contain at least one rule"),
            Self::EmptyRuleId => write!(f, "morphophonological rule id must be non-empty"),
            Self::DuplicateRuleId => write!(f, "morphophonological rule ids must be unique"),
            Self::InvalidMorphology => write!(f, "morphophonological rule morphology contains duplicate feature categories"),
            Self::AmbiguousFeatureMatch => write!(f, "morphophonological rule set contains duplicate exact feature matches"),
            Self::EmptyOperationOperand => write!(f, "morphophonological rule operation operand must be non-empty"),
            Self::EmptyLemma => write!(f, "morphophonological derivation lemma must be non-empty"),
            Self::NoMatchingRule => write!(f, "morphophonological rule set has no rule for the exact morphology"),
            Self::SourceFormMismatch => write!(f, "morphophonological rule operation source does not match the lemma"),
            Self::EmptyDerivedForm => write!(f, "morphophonological rule produced an empty form"),
        }
    }
}

impl std::error::Error for MorphophonologicalRuleSetError {}

/// One explicit derivation step connecting a selected lemma and morphology to the
/// morphophonological form supplied to downstream phonological realization.
///
/// This is an evidence contract, not a language engine: validation proves that the trace
/// exactly describes the retained binding and rule-set identity. It does not prove that the
/// referenced linguistic rule is complete, correct, natural, or appropriate for speakers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalDerivationStep {
    pub position: usize,
    pub lexeme_id: String,
    pub lemma: String,
    pub morphology: Vec<MorphologicalFeature>,
    pub output_form: String,
    pub language_tag: String,
    pub rule_id: String,
    pub rule_provenance: String,
    /// Identifier of the concrete executable rule selected within the rule set.
    pub applied_rule_id: String,
}

/// Explicit, fail-closed evidence that every bound lexical constituent has a recorded
/// morphophonological realization under one explicitly identified language rule set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalDerivationWitness {
    pub version: String,
    pub language: LanguageRuleBinding,
    /// Stable provenance identifier for the executable rule resource.
    pub rule_set_source_id: String,
    /// Explicit scope claim carried by the executable rule resource.
    pub rule_set_dialect_scope: String,
    /// Exact executable rule-selection policy.
    pub rule_set_selection_policy: String,
    /// Typed origin/provenance for the executable resource; absent for caller-supplied
    /// historical witnesses that make no executable resource claim.
    pub rule_set_resource_evidence: Option<MorphophonologicalResourceEvidence>,
    /// Digest of the exact serialized executable rule set used for derivation.
    pub rule_set_blake3: String,
    pub steps: Vec<MorphophonologicalDerivationStep>,
}

impl MorphophonologicalDerivationWitness {
    pub fn new(
        binding: &LexicalMorphosyntacticBinding,
        steps: Vec<MorphophonologicalDerivationStep>,
    ) -> Result<Self, MorphophonologicalDerivationWitnessError> {
        let witness = Self {
            version: MORPHOPHONOLOGICAL_DERIVATION_WITNESS_VERSION.to_string(),
            language: binding.language.clone(),
            rule_set_source_id: String::new(),
            rule_set_dialect_scope: String::new(),
            rule_set_selection_policy: String::new(),
            rule_set_resource_evidence: None,
            rule_set_blake3: String::new(),
            steps,
        };
        witness.validate_against_binding(binding)?;
        Ok(witness)
    }

    /// Derive a witness by executing the supplied deterministic rule set against every
    /// lexical constituent, then require the resulting surface form to equal the binding's
    /// retained morphophonological form.
    pub fn from_rule_set(
        binding: &LexicalMorphosyntacticBinding,
        rule_set: &MorphophonologicalRuleSet,
    ) -> Result<Self, MorphophonologicalDerivationWitnessError> {
        binding
            .validate()
            .map_err(|_| MorphophonologicalDerivationWitnessError::InvalidBinding)?;
        rule_set
            .validate()
            .map_err(|_| MorphophonologicalDerivationWitnessError::InvalidRuleSet)?;

        if binding.language.status != LanguageRuleStatus::Bound
            || binding.language.language_tag != rule_set.language_tag
            || binding.language.rule_id.as_deref() != Some(rule_set.rule_id.as_str())
            || binding.language.provenance.as_deref() != Some(rule_set.provenance.as_str())
        {
            return Err(MorphophonologicalDerivationWitnessError::RuleSetIdentityMismatch);
        }

        let mut steps = Vec::with_capacity(binding.constituents.len());
        for constituent in &binding.constituents {
            let output = constituent
                .morphophonological_form
                .as_deref()
                .ok_or(MorphophonologicalDerivationWitnessError::MissingOutputForm {
                    position: constituent.position,
                })?;

            let (derived_output, applied_rule_id) = rule_set
                .derive(&constituent.lemma, &constituent.morphology)
                .map_err(|_| MorphophonologicalDerivationWitnessError::RuleExecutionFailed {
                    position: constituent.position,
                })?;
            if derived_output != output {
                return Err(MorphophonologicalDerivationWitnessError::DerivedOutputMismatch {
                    position: constituent.position,
                });
            }

            steps.push(MorphophonologicalDerivationStep {
                position: constituent.position,
                lexeme_id: constituent.lexeme_id.clone(),
                lemma: constituent.lemma.clone(),
                morphology: constituent.morphology.clone(),
                output_form: output.to_string(),
                language_tag: binding.language.language_tag.clone(),
                rule_id: rule_set.rule_id.clone(),
                rule_provenance: rule_set.provenance.clone(),
                applied_rule_id,
            });
        }

        let mut witness = Self::new(binding, steps)
            .map_err(|_| MorphophonologicalDerivationWitnessError::InvalidDerivedWitness)?;
        witness.rule_set_source_id = rule_set.source_id.clone();
        witness.rule_set_dialect_scope = rule_set.dialect_scope.clone();
        witness.rule_set_selection_policy = rule_set.selection_policy.clone();
        witness.rule_set_resource_evidence = Some(rule_set.resource_evidence.clone());
        witness.rule_set_blake3 = rule_set.resource_blake3();
        witness.validate_against_binding_and_rule_set(binding, rule_set)?;
        Ok(witness)
    }

    /// Validate the retained trace against the executable rule set as well as the binding.
    pub fn validate_against_binding_and_rule_set(
        &self,
        binding: &LexicalMorphosyntacticBinding,
        rule_set: &MorphophonologicalRuleSet,
    ) -> Result<(), MorphophonologicalDerivationWitnessError> {
        self.validate_against_binding(binding)?;
        rule_set
            .validate()
            .map_err(|_| MorphophonologicalDerivationWitnessError::InvalidRuleSet)?;

        if self.rule_set_source_id != rule_set.source_id
            || self.rule_set_dialect_scope != rule_set.dialect_scope
            || self.rule_set_selection_policy != rule_set.selection_policy
            || self.rule_set_resource_evidence.as_ref() != Some(&rule_set.resource_evidence)
        {
            return Err(MorphophonologicalDerivationWitnessError::RuleSetMetadataMismatch);
        }

        if self.rule_set_blake3.trim().is_empty()
            || self.rule_set_blake3.len() != 64
            || !self.rule_set_blake3.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(MorphophonologicalDerivationWitnessError::MalformedRuleSetDigest);
        }

        if self.rule_set_blake3 != rule_set.resource_blake3() {
            return Err(MorphophonologicalDerivationWitnessError::RuleSetContentMismatch);
        }

        if binding.language.language_tag != rule_set.language_tag
            || binding.language.rule_id.as_deref() != Some(rule_set.rule_id.as_str())
            || binding.language.provenance.as_deref() != Some(rule_set.provenance.as_str())
        {
            return Err(MorphophonologicalDerivationWitnessError::RuleSetIdentityMismatch);
        }

        for (step, constituent) in self.steps.iter().zip(&binding.constituents) {
            let (derived_output, applied_rule_id) = rule_set
                .derive(&constituent.lemma, &constituent.morphology)
                .map_err(|_| MorphophonologicalDerivationWitnessError::RuleExecutionFailed {
                    position: constituent.position,
                })?;
            if derived_output != constituent.morphophonological_form.as_deref().unwrap_or_default()
                || step.output_form != derived_output
            {
                return Err(MorphophonologicalDerivationWitnessError::DerivedOutputMismatch {
                    position: constituent.position,
                });
            }
            if step.applied_rule_id != applied_rule_id {
                return Err(MorphophonologicalDerivationWitnessError::AppliedRuleMismatch {
                    position: constituent.position,
                });
            }
        }

        Ok(())
    }

    /// Validate the complete trace against the exact persisted lexical/morphosyntactic binding.
    pub fn validate_against_binding(
        &self,
        binding: &LexicalMorphosyntacticBinding,
    ) -> Result<(), MorphophonologicalDerivationWitnessError> {
        binding
            .validate()
            .map_err(|_| MorphophonologicalDerivationWitnessError::InvalidBinding)?;

        if self.version != MORPHOPHONOLOGICAL_DERIVATION_WITNESS_VERSION {
            return Err(MorphophonologicalDerivationWitnessError::InvalidVersion);
        }

        if self.language != binding.language {
            return Err(MorphophonologicalDerivationWitnessError::LanguageBindingMismatch);
        }

        if !matches!(binding.language.status, LanguageRuleStatus::Bound) {
            return Err(MorphophonologicalDerivationWitnessError::LanguageRulesMustBeBound);
        }

        if self.steps.len() != binding.constituents.len() {
            return Err(MorphophonologicalDerivationWitnessError::StepCountMismatch);
        }

        let rule_id = binding
            .language
            .rule_id
            .as_deref()
            .ok_or(MorphophonologicalDerivationWitnessError::LanguageRulesMustBeBound)?;
        let rule_provenance = binding
            .language
            .provenance
            .as_deref()
            .ok_or(MorphophonologicalDerivationWitnessError::LanguageRulesMustBeBound)?;

        for (expected_position, (step, constituent)) in
            self.steps.iter().zip(&binding.constituents).enumerate()
        {
            if step.position != expected_position || constituent.position != expected_position {
                return Err(MorphophonologicalDerivationWitnessError::PositionMismatch {
                    position: expected_position,
                });
            }
            if step.lexeme_id != constituent.lexeme_id {
                return Err(MorphophonologicalDerivationWitnessError::LexemeIdentityMismatch {
                    position: expected_position,
                });
            }
            if step.lemma != constituent.lemma {
                return Err(MorphophonologicalDerivationWitnessError::LemmaMismatch {
                    position: expected_position,
                });
            }
            if step.morphology != constituent.morphology {
                return Err(MorphophonologicalDerivationWitnessError::MorphologyMismatch {
                    position: expected_position,
                });
            }
            let expected_output = constituent
                .morphophonological_form
                .as_deref()
                .ok_or(MorphophonologicalDerivationWitnessError::MissingOutputForm {
                    position: expected_position,
                })?;
            if step.output_form != expected_output {
                return Err(MorphophonologicalDerivationWitnessError::OutputFormMismatch {
                    position: expected_position,
                });
            }
            if step.language_tag != binding.language.language_tag {
                return Err(MorphophonologicalDerivationWitnessError::LanguageTagMismatch {
                    position: expected_position,
                });
            }
            if step.rule_id != rule_id {
                return Err(MorphophonologicalDerivationWitnessError::RuleIdMismatch {
                    position: expected_position,
                });
            }
            if step.rule_provenance != rule_provenance {
                return Err(MorphophonologicalDerivationWitnessError::RuleProvenanceMismatch {
                    position: expected_position,
                });
            }
            if step.output_form.trim().is_empty() {
                return Err(MorphophonologicalDerivationWitnessError::EmptyOutputForm {
                    position: expected_position,
                });
            }
        }

        Ok(())
    }

    pub fn grounding_surface(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            format!(
                "{{\"version\":\"{}\",\"serialization\":\"failed\"}}",
                self.version
            )
        })
    }

    pub fn provenance_token(&self) -> String {
        let digest = blake3::hash(self.grounding_surface().as_bytes());
        digest.to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphophonologicalDerivationWitnessError {
    InvalidVersion,
    InvalidBinding,
    LanguageBindingMismatch,
    LanguageRulesMustBeBound,
    StepCountMismatch,
    PositionMismatch { position: usize },
    LexemeIdentityMismatch { position: usize },
    LemmaMismatch { position: usize },
    MorphologyMismatch { position: usize },
    MissingOutputForm { position: usize },
    OutputFormMismatch { position: usize },
    LanguageTagMismatch { position: usize },
    RuleIdMismatch { position: usize },
    RuleProvenanceMismatch { position: usize },
    EmptyOutputForm { position: usize },
    InvalidRuleSet,
    RuleSetIdentityMismatch,
    MalformedRuleSetDigest,
    RuleSetMetadataMismatch,
    RuleSetContentMismatch,
    RuleExecutionFailed { position: usize },
    DerivedOutputMismatch { position: usize },
    AppliedRuleMismatch { position: usize },
    InvalidDerivedWitness,
}
}

impl std::fmt::Display for MorphophonologicalDerivationWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "morphophonological derivation witness version is unsupported"),
            Self::InvalidBinding => write!(f, "lexical binding for morphophonological derivation is invalid"),
            Self::LanguageBindingMismatch => write!(f, "morphophonological witness language binding does not match the lexical binding"),
            Self::LanguageRulesMustBeBound => write!(f, "morphophonological derivation requires explicitly bound language rules"),
            Self::StepCountMismatch => write!(f, "morphophonological derivation step count does not match lexical constituents"),
            Self::PositionMismatch { position } => write!(f, "morphophonological derivation position {position} is not aligned with the lexical binding"),
            Self::LexemeIdentityMismatch { position } => write!(f, "morphophonological derivation lexeme identity mismatches position {position}"),
            Self::LemmaMismatch { position } => write!(f, "morphophonological derivation lemma mismatches position {position}"),
            Self::MorphologyMismatch { position } => write!(f, "morphophonological derivation morphology mismatches position {position}"),
            Self::MissingOutputForm { position } => write!(f, "lexical position {position} has no morphophonological output form"),
            Self::OutputFormMismatch { position } => write!(f, "morphophonological derivation output mismatches position {position}"),
            Self::LanguageTagMismatch { position } => write!(f, "morphophonological derivation language tag mismatches position {position}"),
            Self::RuleIdMismatch { position } => write!(f, "morphophonological derivation rule id mismatches position {position}"),
            Self::RuleProvenanceMismatch { position } => write!(f, "morphophonological derivation rule provenance mismatches position {position}"),
            Self::EmptyOutputForm { position } => write!(f, "morphophonological derivation output at position {position} is empty"),
            Self::InvalidRuleSet => write!(f, "morphophonological derivation rule set is invalid"),
            Self::RuleSetIdentityMismatch => write!(f, "morphophonological derivation rule-set identity does not match the lexical language binding"),
            Self::MalformedRuleSetDigest => write!(f, "morphophonological derivation witness rule-set digest is malformed"),
            Self::RuleSetMetadataMismatch => write!(f, "morphophonological derivation witness rule-set metadata does not match the executable resource"),
            Self::RuleSetContentMismatch => write!(f, "morphophonological derivation witness does not match the exact executable rule-set content"),
            Self::RuleExecutionFailed { position } => write!(f, "morphophonological rule execution failed at position {position}"),
            Self::DerivedOutputMismatch { position } => write!(f, "executable morphophonological derivation does not reproduce position {position}'s retained output"),
            Self::AppliedRuleMismatch { position } => write!(f, "retained morphophonological witness selected a different executable rule at position {position}"),
            Self::InvalidDerivedWitness => write!(f, "executable morphophonological derivation produced an invalid witness"),
        }
    }
}

impl std::error::Error for MorphophonologicalDerivationWitnessError {}

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
        if matches!(frame.strategy, FormulationStrategy::Abstain) {
            return Err(LexicalBindingError::AbstentionCannotBind);
        }

        let binding = Self {
            version: LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION.to_string(),
            source_frame_version: frame.version.clone(),
            source_frame_grounding: frame.canonical_grounding_surface(),
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
        if matches!(frame.strategy, FormulationStrategy::Abstain) {
            return Err(LexicalBindingError::AbstentionCannotBind);
        }
        self.validate()?;

        if self.source_frame_version != frame.version
            || self.source_frame_grounding != frame.canonical_grounding_surface()
        {
            return Err(LexicalBindingError::UpstreamMismatch);
        }

        let expected_sources: HashSet<(String, String)> = frame
            .constituents
            .iter()
            .map(|slot| (slot.role.clone(), slot.prime.clone()))
            .collect();
        let expected_source_order: Vec<(String, String)> = frame
            .constituents
            .iter()
            .map(|slot| (slot.role.clone(), slot.prime.clone()))
            .collect();

        let mut observed_sources = HashSet::new();
        let mut observed_source_order = Vec::new();
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
                    observed_source_order.push((role.clone(), prime.clone()));
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

        if observed_source_order != expected_source_order {
            return Err(LexicalBindingError::SemanticSourceOrderMismatch);
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
    AbstentionCannotBind,
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
    SemanticSourceOrderMismatch,
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
            Self::AbstentionCannotBind => write!(f, "abstaining linguistic frames cannot be lexically bound"),
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
            Self::SemanticSourceOrderMismatch => write!(f, "lexical semantic-source order no longer matches the linguistic frame"),
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
    fn legacy_binding_schema_fails_closed() {
        let frame = statement_frame();
        let mut binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            base_semantic_bindings(),
            vec![],
            vec![],
        )
        .expect("base lexical binding");
        binding.version = "broca-lexical-morphosyntactic-binding-v1".into();

        assert_eq!(
            binding.validate().expect_err("legacy binding schema must fail closed"),
            LexicalBindingError::InvalidVersion
        );
    }

    #[test]
    fn upstream_prosody_tampering_breaks_exact_lineage() {
        let frame = statement_frame();
        let binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            base_semantic_bindings(),
            vec![],
            vec![],
        )
        .expect("base lexical binding");
        let mut tampered = frame.clone();
        tampered.prosody.rate = if tampered.prosody.rate < 1.0 { 1.2 } else { 0.8 };

        assert_eq!(
            binding
                .validate_against_frame(&tampered)
                .expect_err("prosody mutation must invalidate lexical lineage"),
            LexicalBindingError::UpstreamMismatch
        );
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
    fn semantic_source_order_must_match_frame_linearization() {
        let frame = statement_frame();
        let mut bindings = base_semantic_bindings();
        bindings.swap(0, 1);
        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings,
            vec![],
            vec![],
        )
        .expect_err("semantic sources must preserve the already-authorized frame order");

        assert_eq!(error, LexicalBindingError::SemanticSourceOrderMismatch);
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
    fn persisted_binding_cannot_target_abstention_frame() {
        let frame = statement_frame();
        let bindings = base_semantic_bindings();
        let mut binding = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            bindings.clone(),
            vec![],
            vec![],
        )
        .expect("base lexical binding");

        let genesis = GenesisSeed::from_phrase("lexical-binding-test");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(7);
        channels.set_epistemic(3.0);
        let readout = decoder.decode(&channels);
        let mut abstention_frame =
            LinguisticFrame::from_speech_plan(&crate::SpeechPlan::from_readout(&channels, &readout));
        binding.source_frame_version = abstention_frame.version.clone();
        binding.source_frame_grounding = abstention_frame.grounding_surface();

        abstention_frame.strategy = FormulationStrategy::Abstain;

        assert_eq!(
            binding
                .validate_against_frame(&abstention_frame)
                .expect_err("persisted binding must remain non-realizable"),
            LexicalBindingError::AbstentionCannotBind
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
    fn abstention_cannot_be_lexically_bound() {
        let mut frame = statement_frame();
        frame.strategy = FormulationStrategy::Abstain;
        frame.constituents.clear();
        frame.focus_role = None;

        let error = LexicalMorphosyntacticBinding::new(
            &frame,
            english_rule_binding(),
            base_semantic_bindings(),
            vec![],
            vec![],
        )
        .expect_err("abstention must not create lexical binding");

        assert_eq!(error, LexicalBindingError::AbstentionCannotBind);
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

    fn morphophonological_fixture_binding() -> LexicalMorphosyntacticBinding {
        LexicalMorphosyntacticBinding {
            version: LEXICAL_MORPHOSYNTACTIC_BINDING_VERSION.into(),
            source_frame_version: "fixture-frame-v1".into(),
            source_frame_grounding: "fixture-grounding".into(),
            language: LanguageRuleBinding {
                language_tag: "en".into(),
                status: LanguageRuleStatus::Bound,
                rule_id: Some("fixture:english-morphology:v1".into()),
                provenance: Some("fixture:rules:v1".into()),
                unbound_reason: None,
            },
            constituents: vec![LexemeBinding {
                position: 0,
                source: LexicalSource::SemanticConstituent {
                    role: "AGENT".into(),
                    prime: "I".into(),
                },
                lemma: "walk".into(),
                lexeme_id: "fixture:walk".into(),
                grammatical_function: GrammaticalFunction::Verb,
                morphology: vec![MorphologicalFeature {
                    category: "tense".into(),
                    value: "past".into(),
                }],
                morphophonological_form: Some("walked".into()),
                provenance: "fixture:lexicon:v1".into(),
                semantic_payload: true,
            }],
            dependencies: Vec::new(),
            agreement: Vec::new(),
        }
    }

    fn valid_morphophonological_step() -> MorphophonologicalDerivationStep {
        MorphophonologicalDerivationStep {
            position: 0,
            lexeme_id: "fixture:walk".into(),
            lemma: "walk".into(),
            morphology: vec![MorphologicalFeature {
                category: "tense".into(),
                value: "past".into(),
            }],
            output_form: "walked".into(),
            language_tag: "en".into(),
            rule_id: "fixture:english-morphology:v1".into(),
            rule_provenance: "fixture:rules:v1".into(),
            applied_rule_id: "fixture:past-tense".into(),
        }
    }

    fn morphophonological_fixture_rule_set() -> MorphophonologicalRuleSet {
        MorphophonologicalRuleSet::new(
            "en",
            "fixture:english-rules-v1",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "symthaea-fixture-rule-resource",
                "fixture-v1",
            )
            .unwrap(),
            "fixture:english-morphology:v1",
            "fixture:rules:v1",
            vec![MorphophonologicalRule {
                rule_id: "fixture:past-tense".into(),
                morphology: vec![MorphologicalFeature {
                    category: "tense".into(),
                    value: "past".into(),
                }],
                operation: MorphophonologicalRuleOperation::AppendSuffix {
                    suffix: "ed".into(),
                },
            }],
        )
        .expect("fixture rule set must validate")
    }

    #[test]
    fn morphophonological_derivation_witness_binds_exact_inputs_and_rule_identity() {
        let binding = morphophonological_fixture_binding();
        let witness = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .expect("explicit derivation trace should validate");

        assert!(witness.validate_against_binding(&binding).is_ok());
        assert_eq!(witness.provenance_token().len(), 64);
    }

    #[test]
    fn morphophonological_derivation_witness_rejects_output_tampering() {
        let binding = morphophonological_fixture_binding();
        let mut witness = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .unwrap();
        witness.steps[0].output_form = "walks".into();

        assert_eq!(
            witness
                .validate_against_binding(&binding)
                .expect_err("output tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::OutputFormMismatch { position: 0 }
        );
    }

    #[test]
    fn morphophonological_derivation_witness_rejects_lemma_and_morphology_tampering() {
        let binding = morphophonological_fixture_binding();

        let mut lemma_tampered = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .unwrap();
        lemma_tampered.steps[0].lemma = "run".into();
        assert_eq!(
            lemma_tampered
                .validate_against_binding(&binding)
                .expect_err("lemma tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::LemmaMismatch { position: 0 }
        );

        let mut morphology_tampered = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .unwrap();
        morphology_tampered.steps[0].morphology[0].value = "present".into();
        assert_eq!(
            morphology_tampered
                .validate_against_binding(&binding)
                .expect_err("morphology tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::MorphologyMismatch { position: 0 }
        );
    }

    #[test]
    fn morphophonological_derivation_witness_rejects_rule_and_language_tampering() {
        let binding = morphophonological_fixture_binding();

        let mut rule_tampered = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .unwrap();
        rule_tampered.steps[0].rule_id = "fixture:other-rule:v1".into();
        assert_eq!(
            rule_tampered
                .validate_against_binding(&binding)
                .expect_err("rule identity tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::RuleIdMismatch { position: 0 }
        );

        let mut language_tampered = MorphophonologicalDerivationWitness::new(
            &binding,
            vec![valid_morphophonological_step()],
        )
        .unwrap();
        language_tampered.steps[0].language_tag = "en-GB".into();
        assert_eq!(
            language_tampered
                .validate_against_binding(&binding)
                .expect_err("language tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::LanguageTagMismatch { position: 0 }
        );
    }

    #[test]
    fn morphophonological_derivation_witness_rejects_unbound_rule_sets() {
        let mut binding = morphophonological_fixture_binding();
        binding.language = LanguageRuleBinding {
            language_tag: "en".into(),
            status: LanguageRuleStatus::Unbound,
            rule_id: None,
            provenance: None,
            unbound_reason: Some("fixture has no executable rule set".into()),
        };

        let error = MorphophonologicalDerivationWitness::new(&binding, Vec::new())
            .expect_err("unbound language rules must not be elevated to derivation evidence");
        assert_eq!(
            error,
            MorphophonologicalDerivationWitnessError::LanguageRulesMustBeBound
        );
    }

    #[test]
    fn morphophonological_derivation_executes_and_reproduces_binding() {
        let binding = morphophonological_fixture_binding();
        let rule_set = morphophonological_fixture_rule_set();

        let witness = MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
            .expect("executable rule set should derive retained form");

        assert_eq!(witness.steps[0].applied_rule_id, "fixture:past-tense");
        witness
            .validate_against_binding_and_rule_set(&binding, &rule_set)
            .expect("current executable rule set should reproduce witness");
    }

    #[test]
    fn morphophonological_derivation_fails_closed_when_rule_cannot_execute() {
        let binding = morphophonological_fixture_binding();
        let rule_set = MorphophonologicalRuleSet::new(
            "en",
            "fixture:english-rules-v1",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "symthaea-fixture-rule-resource",
                "fixture-v1",
            )
            .unwrap(),
            "fixture:english-morphology:v1",
            "fixture:rules:v1",
            vec![MorphophonologicalRule {
                rule_id: "fixture:past-tense".into(),
                morphology: vec![MorphologicalFeature {
                    category: "tense".into(),
                    value: "past".into(),
                }],
                operation: MorphophonologicalRuleOperation::ReplaceSuffix {
                    from: "y".into(),
                    to: "ied".into(),
                },
            }],
        )
        .unwrap();

        assert_eq!(
            MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
                .expect_err("rule must not claim to derive an inapplicable lemma"),
            MorphophonologicalDerivationWitnessError::RuleExecutionFailed { position: 0 }
        );
    }

    #[test]
    fn morphophonological_derivation_fails_closed_when_retained_output_is_tampered() {
        let mut binding = morphophonological_fixture_binding();
        binding.constituents[0].morphophonological_form = Some("walks".into());
        assert_eq!(
            MorphophonologicalDerivationWitness::from_rule_set(
                &binding,
                &morphophonological_fixture_rule_set(),
            )
            .expect_err("executable derivation must reject retained output tampering"),
            MorphophonologicalDerivationWitnessError::DerivedOutputMismatch { position: 0 }
        );
    }

    #[test]
    fn morphophonological_rule_set_rejects_ambiguous_exact_feature_rules() {
        let error = MorphophonologicalRuleSet::new(
            "en",
            "fixture:ambiguous-rules-v1",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "symthaea-fixture-ambiguous-resource",
                "fixture-v1",
            )
            .unwrap(),
            "fixture:ambiguous:v1",
            "fixture:rules:v1",
            vec![
                MorphophonologicalRule {
                    rule_id: "fixture:one".into(),
                    morphology: vec![MorphologicalFeature {
                        category: "tense".into(),
                        value: "past".into(),
                    }],
                    operation: MorphophonologicalRuleOperation::AppendSuffix {
                        suffix: "ed".into(),
                    },
                },
                MorphophonologicalRule {
                    rule_id: "fixture:two".into(),
                    morphology: vec![MorphologicalFeature {
                        category: "tense".into(),
                        value: "past".into(),
                    }],
                    operation: MorphophonologicalRuleOperation::AppendSuffix {
                        suffix: "t".into(),
                    },
                },
            ],
        )
        .expect_err("duplicate exact feature matches must fail closed");
        assert_eq!(
            error,
            MorphophonologicalRuleSetError::AmbiguousFeatureMatch
        );
    }

    #[test]
    fn morphophonological_rule_set_identity_is_part_of_replay_validation() {
        let binding = morphophonological_fixture_binding();
        let mut rule_set = morphophonological_fixture_rule_set();
        rule_set.provenance = "fixture:other-rules:v1".into();

        assert_eq!(
            MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
                .expect_err("rule-set provenance drift must fail closed"),
            MorphophonologicalDerivationWitnessError::RuleSetIdentityMismatch
        );
    }


    #[test]
    fn morphophonological_rule_set_digest_binds_exact_rule_content() {
        let mut rule_set = morphophonological_fixture_rule_set();
        let before = rule_set.resource_blake3();
        rule_set.rules[0].operation = MorphophonologicalRuleOperation::AppendSuffix {
            suffix: "t".into(),
        };
        assert_ne!(before, rule_set.resource_blake3());
    }

    #[test]
    fn morphophonological_derivation_rejects_rule_content_tampering() {
        let binding = morphophonological_fixture_binding();
        let rule_set = morphophonological_fixture_rule_set();
        let mut witness = MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
            .expect("fixture rule set should derive the binding");
        let mut tampered_rules = rule_set.clone();
        tampered_rules.rules[0].operation = MorphophonologicalRuleOperation::AppendSuffix {
            suffix: "t".into(),
        };

        assert_eq!(
            witness
                .validate_against_binding_and_rule_set(&binding, &tampered_rules)
                .expect_err("exact rule content tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::RuleSetContentMismatch
        );

        witness.rule_set_blake3 = "not-a-digest".into();
        assert_eq!(
            witness
                .validate_against_binding_and_rule_set(&binding, &rule_set)
                .expect_err("malformed rule-set digest must fail closed"),
            MorphophonologicalDerivationWitnessError::MalformedRuleSetDigest
        );
    }


    #[test]
    fn morphophonological_rule_set_metadata_is_part_of_exact_resource_identity() {
        let rule_set = morphophonological_fixture_rule_set();
        let mut tampered = rule_set.clone();
        tampered.source_id = "fixture:other-source-v1".into();
        assert_ne!(rule_set.resource_blake3(), tampered.resource_blake3());

        let mut witness = MorphophonologicalDerivationWitness::from_rule_set(
            &morphophonological_fixture_binding(),
            &rule_set,
        )
        .unwrap();
        witness.rule_set_source_id = tampered.source_id.clone();
        assert_eq!(
            witness
                .validate_against_binding_and_rule_set(
                    &morphophonological_fixture_binding(),
                    &tampered,
                )
                .expect_err("source metadata tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::RuleSetMetadataMismatch
        );

        let mut scope_tampered = rule_set.clone();
        scope_tampered.dialect_scope = "en-GB".into();
        assert_ne!(rule_set.resource_blake3(), scope_tampered.resource_blake3());
        witness.rule_set_source_id = rule_set.source_id.clone();
        witness.rule_set_dialect_scope = scope_tampered.dialect_scope.clone();
        assert_eq!(
            witness
                .validate_against_binding_and_rule_set(
                    &morphophonological_fixture_binding(),
                    &scope_tampered,
                )
                .expect_err("scope metadata tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::RuleSetMetadataMismatch
        );

        let mut policy_tampered = rule_set.clone();
        policy_tampered.selection_policy = "first-match-v0".into();
        assert_eq!(
            policy_tampered
                .validate()
                .expect_err("unsupported selection policy must fail closed"),
            MorphophonologicalRuleSetError::UnsupportedSelectionPolicy
        );
    }


    #[test]
    fn morphophonological_resource_evidence_distinguishes_external_and_hand_authored() {
        let hand = MorphophonologicalResourceEvidence::hand_authored(
            "symthaea-fixture",
            "fixture-v1",
        )
        .expect("hand-authored evidence should validate");
        assert_eq!(hand.origin, MorphophonologicalResourceOrigin::HandAuthored);
        assert!(hand.source_uri.is_none());
        assert!(hand.license.is_none());

        let external = MorphophonologicalResourceEvidence::external(
            "example-external-resource",
            "https://example.invalid/resource",
            "commit-or-release-1",
            "CC-BY-SA-4.0",
        )
        .expect("external evidence should validate");
        assert_eq!(external.origin, MorphophonologicalResourceOrigin::External);
        assert!(external.source_uri.is_some());
        assert!(external.license.is_some());
    }

    #[test]
    fn morphophonological_resource_evidence_rejects_missing_external_metadata() {
        let error = MorphophonologicalResourceEvidence {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.into(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: "fixture:external".into(),
            source_uri: None,
            revision: "fixture-rev".into(),
            license: None,
        }
        .validate()
        .expect_err("external resource evidence must identify locator and license");

        assert_eq!(
            error,
            MorphophonologicalResourceEvidenceError::ExternalMissingUri
        );
    }

    #[test]
    fn morphophonological_rule_set_digest_includes_resource_evidence() {
        let rule_set = morphophonological_fixture_rule_set();
        let mut tampered = rule_set.clone();
        tampered.resource_evidence = MorphophonologicalResourceEvidence::hand_authored(
            "symthaea-other-resource",
            "fixture-v1",
        )
        .unwrap();
        assert_ne!(rule_set.resource_blake3(), tampered.resource_blake3());
    }


    #[test]
    fn morphophonological_rule_set_rejects_dual_source_id_drift() {
        let mut rule_set = morphophonological_fixture_rule_set();
        rule_set.source_id = "fixture:other-source-v1".into();

        assert_eq!(
            rule_set
                .validate()
                .expect_err("rule-set and resource evidence identities must remain aligned"),
            MorphophonologicalRuleSetError::ResourceSourceIdMismatch
        );
    }


}
