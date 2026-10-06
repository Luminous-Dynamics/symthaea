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
    "broca-morphophonological-derivation-witness-v2";

pub const MORPHOPHONOLOGICAL_RULE_SET_VERSION: &str =
    "broca-morphophonological-rule-set-v2";
pub const MORPHOPHONOLOGICAL_RULE_SELECTION_POLICY: &str =
    "exact-lemma-and-feature-single-rule-v2";
pub const MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION: &str =
    "broca-morphophonological-resource-evidence-v2";

pub const MORPHOPHONOLOGICAL_COMPILATION_WITNESS_VERSION: &str =
    "broca-morphophonological-compilation-witness-v3";

/// Whether an executable morphology resource is externally sourced or explicitly
/// authored as a local/fixture resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MorphophonologicalResourceOrigin {
    External,
    HandAuthored,
}

/// Provenance for the linguistic/resource artifact behind an executable rule set.
///
/// External resources require a source URI, revision identifier, declared license, and an
/// exact source-artifact digest. Hand-authored resources intentionally carry no fabricated
/// external attribution. The artifact digest is the cryptographic byte identity; the revision
/// field is retained as a human/source-system version reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalResourceEvidence {
    pub version: String,
    pub origin: MorphophonologicalResourceOrigin,
    pub source_id: String,
    pub source_uri: Option<String>,
    pub revision: String,
    pub license: Option<String>,
    /// Exact BLAKE3 identity of the raw source artifact, when one exists.
    ///
    /// This identifies the source artifact bytes; it does not prove that the executable
    /// rule set is a faithful linguistic derivation of those bytes.
    pub source_artifact_blake3: Option<String>,
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
            source_artifact_blake3: None,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn external(
        source_id: impl Into<String>,
        source_uri: impl Into<String>,
        revision: impl Into<String>,
        license: impl Into<String>,
        source_artifact_blake3: impl Into<String>,
    ) -> Result<Self, MorphophonologicalResourceEvidenceError> {
        let evidence = Self {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.to_string(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: source_id.into(),
            source_uri: Some(source_uri.into()),
            revision: revision.into(),
            license: Some(license.into()),
            source_artifact_blake3: Some(source_artifact_blake3.into()),
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// Construct external evidence directly from the exact source-artifact bytes.
    ///
    /// The artifact digest is computed locally from the supplied bytes rather than accepted
    /// as a caller assertion.
    pub fn external_from_artifact(
        source_id: impl Into<String>,
        source_uri: impl Into<String>,
        revision: impl Into<String>,
        license: impl Into<String>,
        source_artifact: &[u8],
    ) -> Result<Self, MorphophonologicalResourceEvidenceError> {
        let digest = blake3::hash(source_artifact).to_hex().to_string();
        Self::external(source_id, source_uri, revision, license, digest)
    }

    /// Verify a persisted source-artifact identity against the exact bytes available now.
    pub fn verify_source_artifact_bytes(
        &self,
        source_artifact: &[u8],
    ) -> Result<(), MorphophonologicalResourceEvidenceError> {
        let expected = self
            .source_artifact_blake3
            .as_deref()
            .ok_or(
                MorphophonologicalResourceEvidenceError::ExternalMissingSourceArtifactDigest,
            )?;
        let actual = blake3::hash(source_artifact).to_hex().to_string();
        if expected != actual {
            return Err(
                MorphophonologicalResourceEvidenceError::SourceArtifactDigestMismatch,
            );
        }
        Ok(())
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
        if let Some(digest) = self.source_artifact_blake3.as_deref() {
            if !is_canonical_blake3_digest(digest) {
                return Err(
                    MorphophonologicalResourceEvidenceError::MalformedSourceArtifactDigest,
                );
            }
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
                if self.source_artifact_blake3.is_none() {
                    return Err(
                        MorphophonologicalResourceEvidenceError::ExternalMissingSourceArtifactDigest,
                    );
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
    ExternalMissingSourceArtifactDigest,
    MalformedSourceArtifactDigest,
    SourceArtifactDigestMismatch,
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
            Self::ExternalMissingSourceArtifactDigest => write!(f, "external morphophonological resources require an exact source artifact BLAKE3 digest"),
            Self::MalformedSourceArtifactDigest => write!(f, "morphophonological source artifact BLAKE3 digest must be exactly 64 lowercase hexadecimal characters"),
            Self::SourceArtifactDigestMismatch => write!(f, "morphophonological source artifact bytes do not match the declared BLAKE3 digest"),
            Self::HandAuthoredCarriesExternalMetadata => write!(f, "hand-authored morphophonological resources must not carry fabricated external URI or license metadata"),
        }
    }
}

impl std::error::Error for MorphophonologicalResourceEvidenceError {}

pub const UNIMORPH_TSV_SOURCE_FORMAT_VERSION: &str =
    "unimorph-tsv-lemma-form-features-v1";
pub const UNIMORPH_TSV_COMPILER_ID: &str = "symthaea-unimorph-tsv-compiler";
pub const UNIMORPH_TSV_COMPILER_VERSION: &str = "broca-unimorph-tsv-compiler-v1";
pub const UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION: &str =
    env!("SYMTHAEA_UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION");
pub const UNIMORPH_TSV_SOURCE_PARSER_REVISION: &str =
    env!("SYMTHAEA_UNIMORPH_TSV_SOURCE_PARSER_REVISION");
pub const UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION: &str =
    env!("SYMTHAEA_UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION");
pub const UNIMORPH_TSV_NORMALIZATION_POLICY: &str =
    "trim-one-line-ending-sort-feature-tokens-sort-output-rules-v1";
pub const UNIMORPH_TSV_FEATURE_BUNDLE_CATEGORY: &str = "unimorph-bundle";

/// A parsed UniMorph-style source record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphophonologicalSourceRecord {
    pub record_id: String,
    pub lemma: String,
    pub form: String,
    pub feature_bundle: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphophonologicalUnimorphCompilerError {
    InvalidSourceFormat,
    EmptyRecord,
    InvalidColumnCount,
    EmptyLemma,
    EmptyForm,
    EmptyFeatureBundle,
    EmptyFeatureToken,
    DuplicateFeatureToken,
    UnsupportedDerivation,
    AmbiguousSimpleDerivation,
    DuplicateLemmaAndFeatureBundle,
    ResourceEvidence,
    RuleSet(MorphophonologicalRuleSetError),
    CompilationWitness(MorphophonologicalCompilationWitnessError),
}

impl std::fmt::Display for MorphophonologicalUnimorphCompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSourceFormat => write!(f, "unsupported UniMorph TSV source format"),
            Self::EmptyRecord => write!(f, "selected UniMorph source record is empty"),
            Self::InvalidColumnCount => write!(f, "UniMorph source record must contain exactly three tab-separated columns"),
            Self::EmptyLemma => write!(f, "UniMorph source record lemma must be non-empty"),
            Self::EmptyForm => write!(f, "UniMorph source record surface form must be non-empty"),
            Self::EmptyFeatureBundle => write!(f, "UniMorph source record feature bundle must be non-empty"),
            Self::EmptyFeatureToken => write!(f, "UniMorph source feature bundle contains an empty token"),
            Self::DuplicateFeatureToken => write!(f, "UniMorph source feature bundle contains a duplicate token"),
            Self::UnsupportedDerivation => write!(f, "UniMorph source record cannot be represented by the supported deterministic rule operations"),
            Self::AmbiguousSimpleDerivation => write!(f, "UniMorph source record admits multiple supported simple derivations"),
            Self::DuplicateLemmaAndFeatureBundle => write!(f, "UniMorph source compilation contains duplicate lemma and feature-bundle identities"),
            Self::ResourceEvidence => write!(f, "UniMorph source resource evidence could not be constructed"),
            Self::RuleSet(error) => write!(f, "compiled morphophonological rule set is invalid: {error}"),
            Self::CompilationWitness(error) => write!(f, "compiled morphophonological provenance is invalid: {error}"),
        }
    }
}

impl std::error::Error for MorphophonologicalUnimorphCompilerError {}

// BEGIN UNIMORPH_TSV_SOURCE_PARSER_SURFACE_V1
pub fn normalize_unimorph_feature_bundle(
    feature_bundle: &str,
) -> Result<MorphologicalFeature, MorphophonologicalUnimorphCompilerError> {
    let mut tokens = feature_bundle
        .split(';')
        .map(str::trim)
        .collect::<Vec<_>>();
    if tokens.is_empty() || tokens.iter().any(|token| token.is_empty()) {
        return Err(MorphophonologicalUnimorphCompilerError::EmptyFeatureToken);
    }
    tokens.sort_unstable();
    for window in tokens.windows(2) {
        if window[0] == window[1] {
            return Err(MorphophonologicalUnimorphCompilerError::DuplicateFeatureToken);
        }
    }
    Ok(MorphologicalFeature {
        category: UNIMORPH_TSV_FEATURE_BUNDLE_CATEGORY.to_string(),
        value: tokens.join(";"),
    })
}

fn parse_unimorph_source_record(
    record_id: &str,
    bytes: &[u8],
) -> Result<MorphophonologicalSourceRecord, MorphophonologicalUnimorphCompilerError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| MorphophonologicalUnimorphCompilerError::InvalidSourceFormat)?;
    let text = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    if text.trim().is_empty() {
        return Err(MorphophonologicalUnimorphCompilerError::EmptyRecord);
    }
    if text.contains('\n') || text.contains('\r') {
        return Err(MorphophonologicalUnimorphCompilerError::InvalidSourceFormat);
    }
    let columns = text.split('\t').collect::<Vec<_>>();
    if columns.len() != 3 {
        return Err(MorphophonologicalUnimorphCompilerError::InvalidColumnCount);
    }
    let lemma = columns[0].trim();
    let form = columns[1].trim();
    let feature_bundle = columns[2].trim();
    if lemma.is_empty() {
        return Err(MorphophonologicalUnimorphCompilerError::EmptyLemma);
    }
    if form.is_empty() {
        return Err(MorphophonologicalUnimorphCompilerError::EmptyForm);
    }
    if feature_bundle.is_empty() {
        return Err(MorphophonologicalUnimorphCompilerError::EmptyFeatureBundle);
    }
    let normalized = normalize_unimorph_feature_bundle(feature_bundle)?;
    Ok(MorphophonologicalSourceRecord {
        record_id: record_id.to_string(),
        lemma: lemma.to_string(),
        form: form.to_string(),
        feature_bundle: normalized.value,
    })
}
// END UNIMORPH_TSV_SOURCE_PARSER_SURFACE_V1

fn infer_simple_unimorph_operation(
    lemma: &str,
    form: &str,
) -> Result<MorphophonologicalRuleOperation, MorphophonologicalUnimorphCompilerError> {
    if lemma == form {
        return Ok(MorphophonologicalRuleOperation::Identity);
    }

    let mut candidates = Vec::with_capacity(2);
    if let Some(suffix) = form.strip_prefix(lemma) {
        if !suffix.is_empty() {
            candidates.push(MorphophonologicalRuleOperation::AppendSuffix {
                suffix: suffix.to_string(),
            });
        }
    }
    if let Some(prefix) = form.strip_suffix(lemma) {
        if !prefix.is_empty() {
            candidates.push(MorphophonologicalRuleOperation::PrependPrefix {
                prefix: prefix.to_string(),
            });
        }
    }

    if candidates.len() == 1 {
        Ok(candidates.remove(0))
    } else if candidates.len() > 1 {
        Err(MorphophonologicalUnimorphCompilerError::AmbiguousSimpleDerivation)
    } else {
        Err(MorphophonologicalUnimorphCompilerError::UnsupportedDerivation)
    }
}

impl MorphophonologicalRuleSet {
    /// Compile selected UniMorph-style TSV records into this module's intentionally narrow
    /// executable vocabulary. No replacement or alternation rule is guessed from a lemma/form
    /// pair: unsupported transformations fail closed.
    pub fn compile_unimorph_tsv_source(
        language_tag: impl Into<String>,
        dialect_scope: impl Into<String>,
        resource_evidence: MorphophonologicalResourceEvidence,
        rule_id: impl Into<String>,
        provenance: impl Into<String>,
        source_artifact: &[u8],
        source_slices: Vec<MorphophonologicalSourceSlice>,
    ) -> Result<(Self, MorphophonologicalCompilationWitness),
        MorphophonologicalUnimorphCompilerError> {
        resource_evidence
            .validate()
            .map_err(|_| MorphophonologicalUnimorphCompilerError::ResourceEvidence)?;
        if let Some(resource_digest) = resource_evidence.source_artifact_blake3.as_deref() {
            if resource_digest != blake3::hash(source_artifact).to_hex().to_string() {
                return Err(MorphophonologicalUnimorphCompilerError::ResourceEvidence);
            }
        }
        let source_id = resource_evidence.source_id.clone();
        let rule_set_id = rule_id.into();

        if source_slices.is_empty() {
            return Err(MorphophonologicalUnimorphCompilerError::EmptyRecord);
        }

        let mut rules = Vec::with_capacity(source_slices.len());
        let mut identities = HashSet::new();
        for slice in &source_slices {
            let end = slice
                .byte_offset
                .checked_add(slice.byte_length)
                .ok_or(MorphophonologicalUnimorphCompilerError::CompilationWitness(
                    MorphophonologicalCompilationWitnessError::SourceRangeOverflow,
                ))?;
            let bytes = source_artifact
                .get(slice.byte_offset..end)
                .ok_or(MorphophonologicalUnimorphCompilerError::CompilationWitness(
                    MorphophonologicalCompilationWitnessError::SourceSliceOutOfBounds,
                ))?;
            let record = parse_unimorph_source_record(&slice.record_id, bytes)?;
            let identity = (record.lemma.clone(), record.feature_bundle.clone());
            if !identities.insert(identity) {
                return Err(
                    MorphophonologicalUnimorphCompilerError::DuplicateLemmaAndFeatureBundle,
                );
            }
            let operation = infer_simple_unimorph_operation(&record.lemma, &record.form)?;
            rules.push(MorphophonologicalRule {
                rule_id: format!("{rule_set_id}:source:{}", record.record_id),
                source_record_id: Some(record.record_id),
                lemma: Some(record.lemma),
                morphology: vec![MorphologicalFeature {
                    category: UNIMORPH_TSV_FEATURE_BUNDLE_CATEGORY.to_string(),
                    value: record.feature_bundle,
                }],
                operation,
            });
        }

        rules.sort_by(|left, right| {
            (
                left.lemma.as_deref().unwrap_or_default(),
                left.morphology
                    .first()
                    .map(|feature| feature.value.as_str())
                    .unwrap_or_default(),
                left.rule_id.as_str(),
            )
                .cmp(&(
                    right.lemma.as_deref().unwrap_or_default(),
                    right
                        .morphology
                        .first()
                        .map(|feature| feature.value.as_str())
                        .unwrap_or_default(),
                    right.rule_id.as_str(),
                ))
        });

        let rule_set = Self::new(
            language_tag,
            source_id,
            dialect_scope,
            resource_evidence,
            rule_set_id,
            provenance,
            rules,
        )
        .map_err(MorphophonologicalUnimorphCompilerError::RuleSet)?;

        let witness = MorphophonologicalCompilationWitness::new(
            UNIMORPH_TSV_COMPILER_ID,
            UNIMORPH_TSV_COMPILER_VERSION,
            UNIMORPH_TSV_NORMALIZATION_POLICY,
            source_artifact,
            source_slices,
            &rule_set,
        )
        .map_err(MorphophonologicalUnimorphCompilerError::CompilationWitness)?;

        Ok((rule_set, witness))
    }
}

/// One exact byte range selected from a frozen source artifact for compilation.
///
/// The verifier uses the offset/length to recompute the record digest from the actual source
/// artifact rather than trusting a caller-supplied record identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalSourceSlice {
    pub record_id: String,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub record_blake3: String,
}

/// Provenance for the transformation from selected source-artifact records into the exact
/// executable morphophonological rule set.
///
/// This proves byte-level source selection and binds that selection to a named compiler policy
/// and exact output rule-set identity. It deliberately does not claim semantic equivalence to
/// the external linguistic resource.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphophonologicalCompilationWitness {
    pub version: String,
    pub compiler_id: String,
    pub compiler_version: String,
    /// Exact content identity of the current UniMorph compiler source module and its build-time
    /// identity mechanism.
    pub compiler_implementation_revision: Option<String>,
    /// Exact content identity of the accepted source-format parser surface and its build-time
    /// identity mechanism.
    pub source_parser_revision: Option<String>,
    /// Exact content identity of the checked-in crate/workspace manifests, Cargo lockfile, and
    /// pinned Rust toolchain used to build this compiler.
    pub compiler_build_context_revision: Option<String>,
    pub normalization_policy: String,
    pub source_artifact_blake3: String,
    pub source_selection_blake3: String,
    pub source_slices: Vec<MorphophonologicalSourceSlice>,
    pub output_rule_set_blake3: String,
    pub transformation_blake3: String,
}

impl MorphophonologicalCompilationWitness {
    pub fn new(
        compiler_id: impl Into<String>,
        compiler_version: impl Into<String>,
        normalization_policy: impl Into<String>,
        source_artifact: &[u8],
        source_slices: Vec<MorphophonologicalSourceSlice>,
        output_rule_set: &MorphophonologicalRuleSet,
    ) -> Result<Self, MorphophonologicalCompilationWitnessError> {
        output_rule_set
            .validate()
            .map_err(|_| MorphophonologicalCompilationWitnessError::InvalidRuleSet)?;

        let compiler_id = compiler_id.into();
        let compiler_version = compiler_version.into();
        let known_unimorph_compiler =
            compiler_id == UNIMORPH_TSV_COMPILER_ID && compiler_version == UNIMORPH_TSV_COMPILER_VERSION;

        let witness = Self {
            version: MORPHOPHONOLOGICAL_COMPILATION_WITNESS_VERSION.to_string(),
            compiler_id,
            compiler_version,
            compiler_implementation_revision: known_unimorph_compiler
                .then(|| UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION.to_string()),
            source_parser_revision: known_unimorph_compiler
                .then(|| UNIMORPH_TSV_SOURCE_PARSER_REVISION.to_string()),
            compiler_build_context_revision: known_unimorph_compiler
                .then(|| UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION.to_string()),
            normalization_policy: normalization_policy.into(),
            source_artifact_blake3: blake3::hash(source_artifact).to_hex().to_string(),
            source_selection_blake3: String::new(),
            source_slices,
            output_rule_set_blake3: output_rule_set.resource_blake3(),
            transformation_blake3: String::new(),
        };

        let mut witness = witness;
        witness.source_selection_blake3 = witness.compute_source_selection_blake3();
        witness.transformation_blake3 = witness.compute_transformation_blake3();
        witness.validate_shape()?;
        witness.validate_against_source_artifact_and_rule_set(source_artifact, output_rule_set)?;
        Ok(witness)
    }

    pub fn validate_shape(&self) -> Result<(), MorphophonologicalCompilationWitnessError> {
        if self.version != MORPHOPHONOLOGICAL_COMPILATION_WITNESS_VERSION {
            return Err(MorphophonologicalCompilationWitnessError::InvalidVersion);
        }
        for value in [
            self.compiler_id.as_str(),
            self.compiler_version.as_str(),
            self.normalization_policy.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(MorphophonologicalCompilationWitnessError::EmptyMetadata);
            }
        }
        if self.compiler_id == UNIMORPH_TSV_COMPILER_ID {
            if self.compiler_version != UNIMORPH_TSV_COMPILER_VERSION
                || self.compiler_implementation_revision.is_none()
                || self.source_parser_revision.is_none()
                || self.compiler_build_context_revision.is_none()
            {
                return Err(
                    MorphophonologicalCompilationWitnessError::MissingCompilerIdentity,
                );
            }
        }
        for value in [
            self.compiler_implementation_revision.as_deref(),
            self.source_parser_revision.as_deref(),
            self.compiler_build_context_revision.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if value.trim().is_empty() {
                return Err(MorphophonologicalCompilationWitnessError::EmptyMetadata);
            }
        }
        if self.compiler_id == UNIMORPH_TSV_COMPILER_ID {
            for value in [
                self.compiler_implementation_revision.as_deref(),
                self.source_parser_revision.as_deref(),
                self.compiler_build_context_revision.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                if !is_canonical_blake3_digest(value) {
                    return Err(MorphophonologicalCompilationWitnessError::MalformedCompilerIdentity);
                }
            }
        }
        if !is_canonical_blake3_digest(&self.source_artifact_blake3)
            || !is_canonical_blake3_digest(&self.source_selection_blake3)
            || !is_canonical_blake3_digest(&self.output_rule_set_blake3)
            || !is_canonical_blake3_digest(&self.transformation_blake3)
        {
            return Err(MorphophonologicalCompilationWitnessError::MalformedDigest);
        }
        if self.source_slices.is_empty() {
            return Err(MorphophonologicalCompilationWitnessError::EmptySourceSelection);
        }

        let mut ids = HashSet::new();
        let mut ranges = self
            .source_slices
            .iter()
            .map(|slice| {
                if slice.record_id.trim().is_empty()
                    || !is_canonical_blake3_digest(&slice.record_blake3)
                {
                    return Err(MorphophonologicalCompilationWitnessError::MalformedSourceSlice);
                }
                if !ids.insert(slice.record_id.clone()) {
                    return Err(MorphophonologicalCompilationWitnessError::DuplicateSourceRecordId);
                }
                if slice.byte_length == 0 {
                    return Err(MorphophonologicalCompilationWitnessError::EmptySourceSlice);
                }
                let end = slice
                    .byte_offset
                    .checked_add(slice.byte_length)
                    .ok_or(MorphophonologicalCompilationWitnessError::SourceRangeOverflow)?;
                Ok((slice.byte_offset, end))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ranges.sort_unstable();
        for window in ranges.windows(2) {
            if window[0].1 > window[1].0 {
                return Err(MorphophonologicalCompilationWitnessError::OverlappingSourceSlices);
            }
        }
        Ok(())
    }

    fn compute_source_selection_blake3(&self) -> String {
        let serialized = serde_json::to_vec(&self.source_slices)
            .unwrap_or_else(|_| b"serialization-failed".to_vec());
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-morphophonological-source-selection-v1\\0");
        hasher.update(&serialized);
        hasher.finalize().to_hex().to_string()
    }

    fn compute_transformation_blake3(&self) -> String {
        let surface = (
            &self.compiler_id,
            &self.compiler_version,
            &self.compiler_implementation_revision,
            &self.source_parser_revision,
            &self.compiler_build_context_revision,
            &self.normalization_policy,
            &self.source_selection_blake3,
            &self.output_rule_set_blake3,
        );
        let serialized = serde_json::to_vec(&surface)
            .unwrap_or_else(|_| b"serialization-failed".to_vec());
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-morphophonological-compilation-v1\\0");
        hasher.update(&serialized);
        hasher.finalize().to_hex().to_string()
    }

    /// Verify that every selected source slice matches the exact source artifact bytes and that
    /// the declared compilation output is the exact current rule-set identity.
    /// Re-execute the UniMorph TSV compiler from the exact persisted source slices and
    /// compare both the emitted executable rule set and compilation witness with this state.
    ///
    /// This is stronger than checking hashes alone: the current compiler implementation must
    /// reproduce the recorded transformation.
    fn validate_current_unimorph_implementation(
        &self,
    ) -> Result<(), MorphophonologicalCompilationWitnessError> {
        if self.compiler_implementation_revision.as_deref()
            != Some(UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION)
        {
            return Err(
                MorphophonologicalCompilationWitnessError::CompilerImplementationRevisionMismatch,
            );
        }
        if self.source_parser_revision.as_deref() != Some(UNIMORPH_TSV_SOURCE_PARSER_REVISION) {
            return Err(MorphophonologicalCompilationWitnessError::SourceParserRevisionMismatch);
        }
        if self.compiler_build_context_revision.as_deref()
            != Some(UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION)
        {
            return Err(
                MorphophonologicalCompilationWitnessError::CompilerBuildContextRevisionMismatch,
            );
        }
        Ok(())
    }

    pub fn replay_unimorph_tsv_compilation(
        &self,
        source_artifact: &[u8],
        output_rule_set: &MorphophonologicalRuleSet,
    ) -> Result<(), MorphophonologicalCompilationWitnessError> {
        self.validate_shape()?;
        if self.compiler_id != UNIMORPH_TSV_COMPILER_ID
            || self.compiler_version != UNIMORPH_TSV_COMPILER_VERSION
        {
            return Err(MorphophonologicalCompilationWitnessError::UnsupportedCompiler);
        }
        self.validate_current_unimorph_implementation()?;

        let (recompiled_rule_set, recompiled_witness) =
            MorphophonologicalRuleSet::compile_unimorph_tsv_source(
                output_rule_set.language_tag.clone(),
                output_rule_set.dialect_scope.clone(),
                output_rule_set.resource_evidence.clone(),
                output_rule_set.rule_id.clone(),
                output_rule_set.provenance.clone(),
                source_artifact,
                self.source_slices.clone(),
            )
            .map_err(|_| MorphophonologicalCompilationWitnessError::CompilerReplayFailed)?;

        if recompiled_rule_set != *output_rule_set {
            return Err(MorphophonologicalCompilationWitnessError::CompilerReplayMismatch);
        }
        if recompiled_witness != *self {
            return Err(MorphophonologicalCompilationWitnessError::WitnessReplayMismatch);
        }
        Ok(())
    }

    pub fn validate_against_source_artifact_and_rule_set(
        &self,
        source_artifact: &[u8],
        output_rule_set: &MorphophonologicalRuleSet,
    ) -> Result<(), MorphophonologicalCompilationWitnessError> {
        self.validate_shape()?;
        output_rule_set
            .validate()
            .map_err(|_| MorphophonologicalCompilationWitnessError::InvalidRuleSet)?;
        if self.source_artifact_blake3
            != blake3::hash(source_artifact).to_hex().to_string()
        {
            return Err(MorphophonologicalCompilationWitnessError::SourceArtifactMismatch);
        }
        if self.output_rule_set_blake3 != output_rule_set.resource_blake3() {
            return Err(MorphophonologicalCompilationWitnessError::OutputRuleSetMismatch);
        }
        let selected_ids = self
            .source_slices
            .iter()
            .map(|slice| slice.record_id.as_str())
            .collect::<HashSet<_>>();
        let output_ids = output_rule_set
            .rules
            .iter()
            .filter_map(|rule| rule.source_record_id.as_deref())
            .collect::<Vec<_>>();
        if !output_ids.is_empty() {
            let output_id_set = output_ids.iter().copied().collect::<HashSet<_>>();
            if output_ids.len() != self.source_slices.len()
                || output_id_set.len() != output_ids.len()
                || output_id_set != selected_ids
            {
                return Err(
                    MorphophonologicalCompilationWitnessError::SourceRecordRuleMappingMismatch,
                );
            }
        }
        if let Some(resource_digest) = output_rule_set
            .resource_evidence
            .source_artifact_blake3
            .as_deref()
        {
            if resource_digest != self.source_artifact_blake3 {
                return Err(
                    MorphophonologicalCompilationWitnessError::SourceArtifactIdentityMismatch,
                );
            }
        }
        if self.source_selection_blake3 != self.compute_source_selection_blake3() {
            return Err(MorphophonologicalCompilationWitnessError::SourceSelectionMismatch);
        }
        if self.transformation_blake3 != self.compute_transformation_blake3() {
            return Err(MorphophonologicalCompilationWitnessError::TransformationMismatch);
        }

        for slice in &self.source_slices {
            let end = slice
                .byte_offset
                .checked_add(slice.byte_length)
                .ok_or(MorphophonologicalCompilationWitnessError::SourceRangeOverflow)?;
            let bytes = source_artifact
                .get(slice.byte_offset..end)
                .ok_or(MorphophonologicalCompilationWitnessError::SourceSliceOutOfBounds)?;
            let actual = blake3::hash(bytes).to_hex().to_string();
            if actual != slice.record_blake3 {
                return Err(MorphophonologicalCompilationWitnessError::SourceRecordMismatch {
                    record_id: slice.record_id.clone(),
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphophonologicalCompilationWitnessError {
    InvalidVersion,
    InvalidRuleSet,
    EmptyMetadata,
    MalformedDigest,
    EmptySourceSelection,
    MalformedSourceSlice,
    EmptySourceSlice,
    DuplicateSourceRecordId,
    SourceRangeOverflow,
    OverlappingSourceSlices,
    SourceArtifactMismatch,
    SourceRecordRuleMappingMismatch,
    SourceSliceOutOfBounds,
    SourceRecordMismatch { record_id: String },
    SourceArtifactIdentityMismatch,
    SourceSelectionMismatch,
    OutputRuleSetMismatch,
    TransformationMismatch,
    UnsupportedCompiler,
    MissingCompilerIdentity,
    MalformedCompilerIdentity,
    CompilerImplementationRevisionMismatch,
    SourceParserRevisionMismatch,
    CompilerBuildContextRevisionMismatch,
    CompilerReplayFailed,
    CompilerReplayMismatch,
    WitnessReplayMismatch,
}

impl std::fmt::Display for MorphophonologicalCompilationWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "morphophonological compilation witness version is unsupported"),
            Self::InvalidRuleSet => write!(f, "morphophonological compilation witness output rule set is invalid"),
            Self::EmptyMetadata => write!(f, "morphophonological compilation witness compiler metadata must be non-empty"),
            Self::MalformedDigest => write!(f, "morphophonological compilation witness contains a malformed BLAKE3 digest"),
            Self::EmptySourceSelection => write!(f, "morphophonological compilation witness must select at least one source record"),
            Self::MalformedSourceSlice => write!(f, "morphophonological compilation witness contains a malformed source slice"),
            Self::EmptySourceSlice => write!(f, "morphophonological compilation witness source slice must contain at least one byte"),
            Self::DuplicateSourceRecordId => write!(f, "morphophonological compilation witness source record ids must be unique"),
            Self::SourceRangeOverflow => write!(f, "morphophonological compilation witness source range overflows"),
            Self::OverlappingSourceSlices => write!(f, "morphophonological compilation witness source slices must not overlap"),
            Self::SourceArtifactMismatch => write!(f, "morphophonological compilation witness source artifact does not match"),
            Self::SourceRecordRuleMappingMismatch => write!(f, "morphophonological compilation witness selected source records do not map one-to-one to compiled rules"),
            Self::SourceSliceOutOfBounds => write!(f, "morphophonological compilation witness source slice is out of bounds"),
            Self::SourceRecordMismatch { record_id } => write!(f, "morphophonological compilation witness source record {record_id} does not match its exact bytes"),
            Self::SourceArtifactIdentityMismatch => write!(f, "morphophonological compilation witness source artifact identity does not match the output rule-set resource evidence"),
            Self::SourceSelectionMismatch => write!(f, "morphophonological compilation witness source selection digest does not match its selected records"),
            Self::OutputRuleSetMismatch => write!(f, "morphophonological compilation witness output rule-set identity does not match"),
            Self::TransformationMismatch => write!(f, "morphophonological compilation witness transformation digest does not match its declared inputs"),
            Self::UnsupportedCompiler => write!(f, "morphophonological compilation witness compiler implementation is not supported for replay"),
            Self::MissingCompilerIdentity => write!(f, "morphophonological compilation witness is missing the implementation identity required for its declared UniMorph compiler"),
            Self::MalformedCompilerIdentity => write!(f, "morphophonological compilation witness contains a malformed UniMorph compiler identity"),
            Self::CompilerImplementationRevisionMismatch => write!(f, "morphophonological compilation witness compiler implementation revision does not match the current compiler"),
            Self::SourceParserRevisionMismatch => write!(f, "morphophonological compilation witness source parser revision does not match the current parser"),
            Self::CompilerBuildContextRevisionMismatch => write!(f, "morphophonological compilation witness compiler build-context revision does not match the current build context"),
            Self::CompilerReplayFailed => write!(f, "morphophonological compilation witness compiler replay failed"),
            Self::CompilerReplayMismatch => write!(f, "morphophonological compilation witness compiler replay did not reproduce the exact output rule set"),
            Self::WitnessReplayMismatch => write!(f, "morphophonological compilation witness compiler replay did not reproduce the exact witness"),
        }
    }
}

impl std::error::Error for MorphophonologicalCompilationWitnessError {}

fn is_canonical_blake3_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

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
    /// Exact source-record identity for rules compiled from a source artifact.
    ///
    /// Hand-authored/generic rules may leave this unset. Compilation-backed rule sets set it
    /// explicitly, and the compilation witness can then require a one-to-one source mapping.
    pub source_record_id: Option<String>,
    /// Optional exact lemma scope. When present, the rule applies only to that lemma.
    ///
    /// This enables compiled paradigm data where many lemmas share the same feature bundle.
    /// Generic and lemma-scoped candidates are not prioritized implicitly; multiple matches
    /// remain an ambiguity and fail closed.
    pub lemma: Option<String>,
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
            if let Some(source_record_id) = rule.source_record_id.as_deref() {
                if source_record_id.trim().is_empty() {
                    return Err(MorphophonologicalRuleSetError::InvalidSourceRecordId);
                }
            }
            if let Some(lemma) = rule.lemma.as_deref() {
                if lemma.trim().is_empty() {
                    return Err(MorphophonologicalRuleSetError::InvalidLemmaScope);
                }
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

            let canonical = (
                rule.lemma.as_deref().unwrap_or_default().to_string(),
                canonical_morphology(&rule.morphology),
            );
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
            .filter(|rule| rule.lemma.as_deref().map_or(true, |scoped| scoped == lemma))
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
    InvalidSourceRecordId,
    InvalidLemmaScope,
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
            Self::InvalidSourceRecordId => write!(f, "morphophonological rule source-record id must be non-empty"),
            Self::InvalidLemmaScope => write!(f, "morphophonological rule lemma scope must be non-empty"),
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
    pub fn validate_against_binding_and_rule_set_with_source_artifact(
        &self,
        binding: &LexicalMorphosyntacticBinding,
        rule_set: &MorphophonologicalRuleSet,
        source_artifact: &[u8],
    ) -> Result<(), MorphophonologicalDerivationWitnessError> {
        self.validate_against_binding_and_rule_set(binding, rule_set)?;
        rule_set
            .resource_evidence
            .verify_source_artifact_bytes(source_artifact)
            .map_err(|_| {
                MorphophonologicalDerivationWitnessError::SourceArtifactDigestMismatch
            })?;
        Ok(())
    }

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
    SourceArtifactDigestMismatch,
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
            Self::SourceArtifactDigestMismatch => write!(f, "morphophonological witness source-artifact bytes do not match the declared resource identity"),
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
            source_record_id: None,
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
                source_record_id: None,
                lemma: None,
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
    fn morphophonological_rule_selection_binds_exact_lemma_scope() {
        let binding = morphophonological_fixture_binding();
        let rule_set = MorphophonologicalRuleSet::new(
            "en",
            "fixture:lemma-scoped-rules-v2",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:lemma-scoped-rules-v2",
                "fixture-v2",
            )
            .unwrap(),
            "fixture:english-morphology:v2",
            "fixture:rules:v2",
            vec![
                MorphophonologicalRule {
                    rule_id: "fixture:walk-past".into(),
                    source_record_id: None,
                    lemma: Some("walk".into()),
                    morphology: vec![MorphologicalFeature {
                        category: "tense".into(),
                        value: "past".into(),
                    }],
                    operation: MorphophonologicalRuleOperation::AppendSuffix {
                        suffix: "ed".into(),
                    },
                },
                MorphophonologicalRule {
                    rule_id: "fixture:jump-past".into(),
                    source_record_id: None,
                    lemma: Some("jump".into()),
                    morphology: vec![MorphologicalFeature {
                        category: "tense".into(),
                        value: "past".into(),
                    }],
                    operation: MorphophonologicalRuleOperation::AppendSuffix {
                        suffix: "ed".into(),
                    },
                },
            ],
        )
        .unwrap();

        assert_eq!(
            rule_set.derive("walk", &binding.constituents[0].morphology).unwrap(),
            ("walked".into(), "fixture:walk-past".into())
        );
        assert_eq!(
            rule_set.derive(
                "jump",
                &binding.constituents[0].morphology
            ).unwrap(),
            ("jumped".into(), "fixture:jump-past".into())
        );
        assert_eq!(
            rule_set
                .derive("run", &binding.constituents[0].morphology)
                .expect_err("unscoped lemma must not borrow another lemma's rule"),
            MorphophonologicalRuleSetError::NoMatchingRule
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
                source_record_id: None,
                lemma: None,
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
    fn morphophonological_rule_set_rejects_duplicate_lemma_and_feature_scope() {
        let error = MorphophonologicalRuleSet::new(
            "en",
            "fixture:duplicate-lemma-scope",
            "en-US",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:duplicate-lemma-scope",
                "fixture-v2",
            )
            .unwrap(),
            "fixture:duplicate-scope:v2",
            "fixture:rules:v2",
            vec![
                MorphophonologicalRule {
                    rule_id: "fixture:walk-one".into(),
                    source_record_id: None,
                    lemma: Some("walk".into()),
                    morphology: vec![MorphologicalFeature {
                        category: "tense".into(),
                        value: "past".into(),
                    }],
                    operation: MorphophonologicalRuleOperation::AppendSuffix {
                        suffix: "ed".into(),
                    },
                },
                MorphophonologicalRule {
                    rule_id: "fixture:walk-two".into(),
                    source_record_id: None,
                    lemma: Some("walk".into()),
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
        .expect_err("duplicate exact lemma and feature scope must fail closed");

        assert_eq!(
            error,
            MorphophonologicalRuleSetError::AmbiguousFeatureMatch
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
                    source_record_id: None,
                    lemma: None,
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
                    source_record_id: None,
                    lemma: None,
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
            &"0".repeat(64),
        )
        .expect("external evidence should validate");
        assert_eq!(external.origin, MorphophonologicalResourceOrigin::External);
        assert!(external.source_uri.is_some());
        assert!(external.license.is_some());
        assert_eq!(
            external.source_artifact_blake3.as_deref(),
            Some("0".repeat(64).as_str())
        );
    }

    #[test]
    fn morphophonological_compilation_witness_binds_source_selection_and_output_rule_set() {
        let binding = morphophonological_fixture_binding();
        let rule_set = morphophonological_fixture_rule_set();
        let artifact = b"row0\tfixture\nrow1\tfixture\n";
        let slices = vec![
            MorphophonologicalSourceSlice {
                record_id: "row0".into(),
                byte_offset: 0,
                byte_length: 13,
                record_blake3: blake3::hash(&artifact[..13]).to_hex().to_string(),
            },
            MorphophonologicalSourceSlice {
                record_id: "row1".into(),
                byte_offset: 13,
                byte_length: artifact.len() - 13,
                record_blake3: blake3::hash(&artifact[13..]).to_hex().to_string(),
            },
        ];

        let witness = MorphophonologicalCompilationWitness::new(
            "symthaea-fixture-compiler",
            "fixture-compiler-v1",
            "fixture-normalization-v1",
            artifact,
            slices,
            &rule_set,
        )
        .expect("compilation witness should validate");

        witness
            .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
            .expect("exact source selection and rule-set identity should replay");

        assert!(witness.compiler_implementation_revision.is_none());
        assert!(witness.source_parser_revision.is_none());
        assert!(witness.compiler_build_context_revision.is_none());

        let mut tampered = artifact.to_vec();
        tampered[0] = b'X';
        assert_eq!(
            witness
                .validate_against_source_artifact_and_rule_set(&tampered, &rule_set)
                .expect_err("source artifact tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::SourceArtifactMismatch
        );

        let mut selection_tampered = witness.clone();
        selection_tampered.source_selection_blake3 = "0".repeat(64);
        assert_eq!(
            selection_tampered
                .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
                .expect_err("source selection digest tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::SourceSelectionMismatch
        );

        let mut transformation_tampered = witness.clone();
        transformation_tampered.transformation_blake3 = "1".repeat(64);
        assert_eq!(
            transformation_tampered
                .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
                .expect_err("transformation digest tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::TransformationMismatch
        );

        let mut slice_tampered = witness.clone();
        slice_tampered.source_slices[0].byte_offset = artifact.len() + 1;
        assert_eq!(
            slice_tampered
                .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
                .expect_err("out-of-bounds source selection must fail closed"),
            MorphophonologicalCompilationWitnessError::SourceSliceOutOfBounds
        );

        let mut rule_tampered = rule_set.clone();
        rule_tampered.rules[0].operation =
            MorphophonologicalRuleOperation::AppendSuffix { suffix: "t".into() };
        assert_eq!(
            witness
                .validate_against_source_artifact_and_rule_set(artifact, &rule_tampered)
                .expect_err("output rule-set tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::OutputRuleSetMismatch
        );
    }

    #[test]
    fn compilation_witness_requires_identity_for_declared_unimorph_compiler() {
        let rule_set = morphophonological_fixture_rule_set();
        let artifact = b"row0	fixture
";
        let witness = MorphophonologicalCompilationWitness {
            version: MORPHOPHONOLOGICAL_COMPILATION_WITNESS_VERSION.into(),
            compiler_id: UNIMORPH_TSV_COMPILER_ID.into(),
            compiler_version: "wrong-version".into(),
            compiler_implementation_revision: None,
            source_parser_revision: None,
            compiler_build_context_revision: None,
            normalization_policy: "fixture-normalization-v1".into(),
            source_artifact_blake3: blake3::hash(artifact).to_hex().to_string(),
            source_selection_blake3: String::new(),
            source_slices: vec![MorphophonologicalSourceSlice {
                record_id: "row0".into(),
                byte_offset: 0,
                byte_length: artifact.len(),
                record_blake3: blake3::hash(artifact).to_hex().to_string(),
            }],
            output_rule_set_blake3: rule_set.resource_blake3(),
            transformation_blake3: String::new(),
        };
        assert_eq!(
            witness.validate_shape().expect_err("declared UniMorph compiler must carry identity"),
            MorphophonologicalCompilationWitnessError::MissingCompilerIdentity
        );
    }

    #[test]
    fn compilation_witness_requires_build_context_identity_for_declared_unimorph_compiler() {
        let rule_set = morphophonological_fixture_rule_set();
        let artifact = b"row0\n";
        let mut witness = MorphophonologicalCompilationWitness::new(
            UNIMORPH_TSV_COMPILER_ID,
            UNIMORPH_TSV_COMPILER_VERSION,
            "fixture-normalization-v1",
            artifact,
            vec![MorphophonologicalSourceSlice {
                record_id: "row0".into(),
                byte_offset: 0,
                byte_length: artifact.len(),
                record_blake3: blake3::hash(artifact).to_hex().to_string(),
            }],
            &rule_set,
        )
        .expect("known UniMorph compiler witness");

        witness.compiler_build_context_revision = None;
        assert_eq!(
            witness
                .validate_shape()
                .expect_err("declared UniMorph compiler must carry build-context identity"),
            MorphophonologicalCompilationWitnessError::MissingCompilerIdentity
        );
    }

    #[test]
    fn compilation_witness_rejects_malformed_unimorph_identity_encoding() {
        let rule_set = morphophonological_fixture_rule_set();
        let artifact = b"row0\n";
        let witness = MorphophonologicalCompilationWitness::new(
            UNIMORPH_TSV_COMPILER_ID,
            UNIMORPH_TSV_COMPILER_VERSION,
            "fixture-normalization-v1",
            artifact,
            vec![MorphophonologicalSourceSlice {
                record_id: "row0".into(),
                byte_offset: 0,
                byte_length: artifact.len(),
                record_blake3: blake3::hash(artifact).to_hex().to_string(),
            }],
            &rule_set,
        )
        .expect("known UniMorph compiler witness");

        let mut tampered = witness;
        tampered.compiler_implementation_revision = Some("not-a-digest".into());
        assert_eq!(
            tampered
                .validate_shape()
                .expect_err("UniMorph compiler identity must be canonical BLAKE3"),
            MorphophonologicalCompilationWitnessError::MalformedCompilerIdentity
        );
    }

    #[test]
    fn unimorph_tsv_compiler_replays_supported_rows_and_normalization() {
        let artifact = b"walk\twalked\tV;PST\ncat\tcat\tN;SG\n";
        let walk_len = b"walk\twalked\tV;PST\n".len();
        let slices = vec![
            MorphophonologicalSourceSlice {
                record_id: "walk-past".into(),
                byte_offset: 0,
                byte_length: walk_len,
                record_blake3: blake3::hash(&artifact[..walk_len]).to_hex().to_string(),
            },
            MorphophonologicalSourceSlice {
                record_id: "cat-singular".into(),
                byte_offset: walk_len,
                byte_length: artifact.len() - walk_len,
                record_blake3: blake3::hash(&artifact[walk_len..]).to_hex().to_string(),
            },
        ];
        let evidence = MorphophonologicalResourceEvidence::hand_authored(
            "fixture:unimorph-tsv",
            "fixture-snapshot-v1",
        )
        .unwrap();

        let (rule_set, witness) = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
            "en",
            "en-unspecified",
            evidence,
            "fixture:unimorph:v1",
            "fixture:compiler:v1",
            artifact,
            slices.clone(),
        )
        .expect("supported UniMorph-style rows should compile");

        assert_eq!(
            rule_set.derive(
                "walk",
                &[MorphologicalFeature {
                    category: UNIMORPH_TSV_FEATURE_BUNDLE_CATEGORY.into(),
                    value: "PST;V".into(),
                }],
            )
            .unwrap(),
            ("walked".into(), "fixture:unimorph:v1:source:walk-past".into())
        );
        assert_eq!(
            rule_set.derive(
                "cat",
                &[MorphologicalFeature {
                    category: UNIMORPH_TSV_FEATURE_BUNDLE_CATEGORY.into(),
                    value: "N;SG".into(),
                }],
            )
            .unwrap(),
            ("cat".into(), "fixture:unimorph:v1:source:cat-singular".into())
        );

        witness
            .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
            .expect("compiler witness should replay against exact source bytes");
        witness
            .replay_unimorph_tsv_compilation(artifact, &rule_set)
            .expect("current UniMorph compiler must reproduce the exact rule set and witness");

        let mut compiler_revision_tampered = witness.clone();
        compiler_revision_tampered.compiler_implementation_revision =
            Some("0".repeat(64));
        assert_eq!(
            compiler_revision_tampered
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("compiler implementation revision tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::CompilerImplementationRevisionMismatch
        );

        let mut parser_revision_tampered = witness.clone();
        parser_revision_tampered.source_parser_revision =
            Some("1".repeat(64));
        assert_eq!(
            parser_revision_tampered
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("source parser revision tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::SourceParserRevisionMismatch
        );

        let mut build_context_tampered = witness.clone();
        build_context_tampered.compiler_build_context_revision =
            Some("2".repeat(64));
        assert_eq!(
            build_context_tampered
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("build-context revision tampering must fail closed"),
            MorphophonologicalCompilationWitnessError::CompilerBuildContextRevisionMismatch
        );

        let mut build_context_recomputed = witness.clone();
        build_context_recomputed.compiler_build_context_revision =
            Some("2".repeat(64));
        build_context_recomputed.transformation_blake3 =
            build_context_recomputed.compute_transformation_blake3();
        assert_eq!(
            build_context_recomputed
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("recomputing the commitment must not self-certify a build context revision"),
            MorphophonologicalCompilationWitnessError::CompilerBuildContextRevisionMismatch
        );

        let mut compiler_revision_recomputed = witness.clone();
        compiler_revision_recomputed.compiler_implementation_revision =
            Some("0".repeat(64));
        compiler_revision_recomputed.transformation_blake3 =
            compiler_revision_recomputed.compute_transformation_blake3();
        assert_eq!(
            compiler_revision_recomputed
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("recomputing the commitment must not self-certify a compiler revision"),
            MorphophonologicalCompilationWitnessError::CompilerImplementationRevisionMismatch
        );

        let mut parser_revision_recomputed = witness.clone();
        parser_revision_recomputed.source_parser_revision =
            Some("1".repeat(64));
        parser_revision_recomputed.transformation_blake3 =
            parser_revision_recomputed.compute_transformation_blake3();
        assert_eq!(
            parser_revision_recomputed
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("recomputing the commitment must not self-certify a parser revision"),
            MorphophonologicalCompilationWitnessError::SourceParserRevisionMismatch
        );

        let mut historical_revision = witness.clone();
        historical_revision.compiler_implementation_revision =
            Some("3".repeat(64));
        historical_revision.transformation_blake3 =
            historical_revision.compute_transformation_blake3();
        historical_revision
            .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
            .expect("historical structural validation should remain inspectable");
        assert_eq!(
            historical_revision
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("current replay must reject a historical implementation revision"),
            MorphophonologicalCompilationWitnessError::CompilerImplementationRevisionMismatch
        );

        let mut replay_tampered = witness.clone();
        replay_tampered.normalization_policy = "different-normalization-v0".into();
        assert_eq!(
            replay_tampered
                .replay_unimorph_tsv_compilation(artifact, &rule_set)
                .expect_err("transformation metadata tampering must fail compiler replay"),
            MorphophonologicalCompilationWitnessError::TransformationMismatch
        );

        let mut reversed = slices;
        reversed.reverse();
        let (reordered_rule_set, reordered_witness) =
            MorphophonologicalRuleSet::compile_unimorph_tsv_source(
                "en",
                "en-unspecified",
                MorphophonologicalResourceEvidence::hand_authored(
                    "fixture:unimorph-tsv",
                    "fixture-snapshot-v1",
                )
                .unwrap(),
                "fixture:unimorph:v1",
                "fixture:compiler:v1",
                artifact,
                reversed,
            )
            .expect("source-record order must not change normalized executable output");
        assert_eq!(rule_set.resource_blake3(), reordered_rule_set.resource_blake3());
        assert_ne!(
            witness.source_selection_blake3,
            reordered_witness.source_selection_blake3
        );
    }

    #[test]
    fn unimorph_tsv_compiler_rejects_embedded_line_endings() {
        let error = parse_unimorph_source_record(
            "embedded-newline",
            b"walk\nwalked\twalked\tV;PST",
        )
        .expect_err("embedded newline must not become part of a source field");
        assert_eq!(
            error,
            MorphophonologicalUnimorphCompilerError::InvalidSourceFormat
        );

        let error = parse_unimorph_source_record(
            "embedded-carriage-return",
            b"walked\twalked\tV;PST\rjunk",
        )
        .expect_err("embedded carriage return must fail closed");
        assert_eq!(
            error,
            MorphophonologicalUnimorphCompilerError::InvalidSourceFormat
        );
    }

    #[test]
    fn unimorph_tsv_compiler_rejects_duplicate_feature_tokens() {
        assert_eq!(
            normalize_unimorph_feature_bundle("V;PST;V")
                .expect_err("duplicate feature token must fail closed"),
            MorphophonologicalUnimorphCompilerError::DuplicateFeatureToken
        );
    }

    #[test]
    fn unimorph_tsv_compiler_rejects_duplicate_lemma_feature_identity() {
        let artifact = b"walk\twalked\tV;PST\nwalk\twalkt\tV;PST\n";
        let first_len = b"walk\twalked\tV;PST\n".len();
        let slices = vec![
            MorphophonologicalSourceSlice {
                record_id: "walk-past-a".into(),
                byte_offset: 0,
                byte_length: first_len,
                record_blake3: blake3::hash(&artifact[..first_len]).to_hex().to_string(),
            },
            MorphophonologicalSourceSlice {
                record_id: "walk-past-b".into(),
                byte_offset: first_len,
                byte_length: artifact.len() - first_len,
                record_blake3: blake3::hash(&artifact[first_len..]).to_hex().to_string(),
            },
        ];

        let error = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
            "en",
            "en-unspecified",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:unimorph-tsv",
                "fixture-snapshot-v1",
            )
            .unwrap(),
            "fixture:unimorph:v1",
            "fixture:compiler:v1",
            artifact,
            slices,
        )
        .expect_err("duplicate lemma and feature identity must fail closed");

        assert_eq!(
            error,
            MorphophonologicalUnimorphCompilerError::DuplicateLemmaAndFeatureBundle
        );
    }

    #[test]
    fn unimorph_tsv_compiler_preserves_external_resource_origin() {
        let artifact = b"walk\twalked\tV;PST\n";
        let evidence = MorphophonologicalResourceEvidence::external_from_artifact(
            "fixture:external-unimorph",
            "https://example.invalid/unimorph/eng",
            "fixture-release-v1",
            "CC-BY-SA-3.0",
            artifact,
        )
        .unwrap();

        let (rule_set, witness) = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
            "en",
            "en-unspecified",
            evidence,
            "fixture:unimorph:v1",
            "fixture:compiler:v1",
            artifact,
            vec![MorphophonologicalSourceSlice {
                record_id: "walk-past".into(),
                byte_offset: 0,
                byte_length: artifact.len(),
                record_blake3: blake3::hash(artifact).to_hex().to_string(),
            }],
        )
        .expect("external evidence should pass through the compiler unchanged");

        assert_eq!(
            rule_set.resource_evidence.origin,
            MorphophonologicalResourceOrigin::External
        );
        witness
            .validate_against_source_artifact_and_rule_set(artifact, &rule_set)
            .expect("external artifact evidence should replay");
    }

    #[test]
    fn unimorph_compilation_witness_rejects_source_to_rule_mapping_drift() {
        let artifact = b"walk\twalked\tV;PST\n";
        let (rule_set, witness) = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
            "en",
            "en-unspecified",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:unimorph-tsv",
                "fixture-snapshot-v1",
            )
            .unwrap(),
            "fixture:unimorph:v1",
            "fixture:compiler:v1",
            artifact,
            vec![MorphophonologicalSourceSlice {
                record_id: "walk-past".into(),
                byte_offset: 0,
                byte_length: artifact.len(),
                record_blake3: blake3::hash(artifact).to_hex().to_string(),
            }],
        )
        .expect("compiler fixture");

        let mut tampered = rule_set;
        tampered.rules[0].source_record_id = Some("different-record".into());
        assert_eq!(
            witness
                .validate_against_source_artifact_and_rule_set(artifact, &tampered)
                .expect_err("source-to-rule mapping drift must fail closed"),
            MorphophonologicalCompilationWitnessError::OutputRuleSetMismatch
        );
    }

    #[test]
    fn unimorph_tsv_compiler_rejects_unsupported_alternation_instead_of_guessing() {
        let artifact = b"study\tstudies\tV;PRS\n";
        let slices = vec![MorphophonologicalSourceSlice {
            record_id: "study-3sg".into(),
            byte_offset: 0,
            byte_length: artifact.len(),
            record_blake3: blake3::hash(artifact).to_hex().to_string(),
        }];

        let error = MorphophonologicalRuleSet::compile_unimorph_tsv_source(
            "en",
            "en-unspecified",
            MorphophonologicalResourceEvidence::hand_authored(
                "fixture:unimorph-tsv",
                "fixture-snapshot-v1",
            )
            .unwrap(),
            "fixture:unimorph:v1",
            "fixture:compiler:v1",
            artifact,
            slices,
        )
        .expect_err("unsupported y->ies alternation must fail closed");

        assert_eq!(
            error,
            MorphophonologicalUnimorphCompilerError::UnsupportedDerivation
        );
    }

    #[test]
    fn morphophonological_resource_evidence_verifies_exact_source_artifact_bytes() {
        let artifact = b"fixture morphology resource v2\nwalk<TAB>walked<TAB>V;PST\n";
        let evidence = MorphophonologicalResourceEvidence::external_from_artifact(
            "fixture:external",
            "https://example.invalid/resource",
            "fixture-release-1",
            "CC-BY-SA-4.0",
            artifact,
        )
        .expect("artifact-derived external evidence should validate");

        evidence
            .verify_source_artifact_bytes(artifact)
            .expect("exact source bytes should match their persisted identity");

        let mut tampered = artifact.to_vec();
        tampered.push(b'!');
        assert_eq!(
            evidence
                .verify_source_artifact_bytes(&tampered)
                .expect_err("source-byte tampering must fail closed"),
            MorphophonologicalResourceEvidenceError::SourceArtifactDigestMismatch
        );
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
            source_artifact_blake3: None,
        }
        .validate()
        .expect_err("external resource evidence must identify locator and license");

        assert_eq!(
            error,
            MorphophonologicalResourceEvidenceError::ExternalMissingUri
        );

        let error = MorphophonologicalResourceEvidence {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.into(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: "fixture:external".into(),
            source_uri: Some("https://example.invalid/resource".into()),
            revision: "fixture-rev".into(),
            license: Some("CC-BY-SA-4.0".into()),
            source_artifact_blake3: None,
        }
        .validate()
        .expect_err("external resource evidence must identify exact artifact bytes");

        assert_eq!(
            error,
            MorphophonologicalResourceEvidenceError::ExternalMissingSourceArtifactDigest
        );
    }

    #[test]
    fn morphophonological_resource_evidence_rejects_malformed_source_artifact_digest() {
        let error = MorphophonologicalResourceEvidence {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.into(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: "fixture:external".into(),
            source_uri: Some("https://example.invalid/resource".into()),
            revision: "fixture-rev".into(),
            license: Some("CC-BY-SA-4.0".into()),
            source_artifact_blake3: Some("A".repeat(64)),
        }
        .validate()
        .expect_err("uppercase hexadecimal must not create multiple digest serializations");

        assert_eq!(
            error,
            MorphophonologicalResourceEvidenceError::MalformedSourceArtifactDigest
        );
    }

    #[test]
    fn morphophonological_witness_verifies_persisted_source_artifact_identity() {
        let binding = morphophonological_fixture_binding();
        let mut rule_set = morphophonological_fixture_rule_set();
        let artifact = b"fixture hand-authored resource bytes";
        rule_set.resource_evidence.source_artifact_blake3 =
            Some(blake3::hash(artifact).to_hex().to_string());
        rule_set.validate().unwrap();

        let witness = MorphophonologicalDerivationWitness::from_rule_set(&binding, &rule_set)
            .expect("artifact-bound rule set should produce a witness");

        witness
            .validate_against_binding_and_rule_set_with_source_artifact(
                &binding,
                &rule_set,
                artifact,
            )
            .expect("matching source artifact bytes should validate");

        let tampered = b"fixture hand-authored resource bytes!";
        assert_eq!(
            witness
                .validate_against_binding_and_rule_set_with_source_artifact(
                    &binding,
                    &rule_set,
                    tampered,
                )
                .expect_err("source artifact tampering must fail closed"),
            MorphophonologicalDerivationWitnessError::SourceArtifactDigestMismatch
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

        let mut external_tampered = morphophonological_fixture_rule_set();
        external_tampered.resource_evidence = MorphophonologicalResourceEvidence {
            version: MORPHOPHONOLOGICAL_RESOURCE_EVIDENCE_VERSION.into(),
            origin: MorphophonologicalResourceOrigin::External,
            source_id: external_tampered.source_id.clone(),
            source_uri: Some("https://example.invalid/resource".into()),
            revision: "fixture-rev".into(),
            license: Some("CC-BY-SA-4.0".into()),
            source_artifact_blake3: Some("0".repeat(64)),
        };
        external_tampered.validate().unwrap();
        let before = external_tampered.resource_blake3();
        external_tampered.resource_evidence.source_artifact_blake3 =
            Some("1".repeat(64));
        assert_ne!(before, external_tampered.resource_blake3());
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



#[test]
fn compilation_witness_rejects_empty_source_slice() {
    let artifact = b"row0\n";
    let rule_set = MorphophonologicalRuleSet::new(
        "en",
        "fixture:rules:v1",
        "en-US",
        MorphophonologicalResourceEvidence::hand_authored(
            "fixture:rules:v1",
            "fixture-v1",
        )
        .expect("fixture resource evidence"),
        "fixture:rules:v1",
        "fixture:rules:v1",
        vec![MorphophonologicalRule {
            rule_id: "fixture:identity".into(),
            source_record_id: Some("row0".into()),
            lemma: Some("row0".into()),
            morphology: vec![MorphologicalFeature {
                category: "fixture".into(),
                value: "identity".into(),
            }],
            operation: MorphophonologicalRuleOperation::Identity,
        }],
    )
    .expect("rule set");

    let witness = MorphophonologicalCompilationWitness::new(
        "fixture-compiler",
        "fixture-compiler-v1",
        "fixture-normalization-v1",
        artifact,
        vec![MorphophonologicalSourceSlice {
            record_id: "row0".into(),
            byte_offset: 0,
            byte_length: artifact.len(),
            record_blake3: blake3::hash(artifact).to_hex().to_string(),
        }],
        &rule_set,
    )
    .expect("compilation witness");

    let mut tampered = witness;
    tampered.source_slices[0].byte_length = 0;
    tampered.source_slices[0].record_blake3 =
        blake3::hash(b"").to_hex().to_string();

    assert_eq!(
        tampered.validate_shape(),
        Err(MorphophonologicalCompilationWitnessError::EmptySourceSlice)
    );
}

}

