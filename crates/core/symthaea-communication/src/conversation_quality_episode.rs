// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-turn conversation-quality episode scorecard.
//!
//! This module composes independent evidence families without creating a
//! universal conversation-quality score. Evidence completeness is deliberately
//! separate from semantic/behavioral quality.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONVERSATION_QUALITY_EPISODE_SCHEMA_V1: &str =
    "symthaea.communication.conversation-quality-episode.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConversationComponentKindV1 {
    FullDuplexVoice,
    Correction,
    PersonaContinuity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentEvidenceStateV1 {
    NotEstablished,
    Partial,
    Established,
    Uncertain,
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentReceiptRefV1 {
    pub kind: ConversationComponentKindV1,
    pub source_head: String,
    pub receipt_schema: String,
    pub receipt_commitment: String,
    pub evidence_state: ComponentEvidenceStateV1,
}

impl ComponentReceiptRefV1 {
    pub fn new(
        kind: ConversationComponentKindV1,
        source_head: impl Into<String>,
        receipt_schema: impl Into<String>,
        receipt_commitment: impl Into<String>,
        evidence_state: ComponentEvidenceStateV1,
    ) -> Result<Self, ConversationQualityEpisodeErrorV1> {
        let source_head = source_head.into();
        if !is_git_sha40(&source_head) {
            return Err(ConversationQualityEpisodeErrorV1::InvalidSourceHead);
        }
        let receipt_schema = canonical_key(receipt_schema.into())?;
        let receipt_commitment = receipt_commitment.into();
        if !is_blake3_commitment(&receipt_commitment) {
            return Err(ConversationQualityEpisodeErrorV1::InvalidReceiptCommitment);
        }
        Ok(Self {
            kind,
            source_head,
            receipt_schema,
            receipt_commitment,
            evidence_state,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationQualityEpisodeManifestV1 {
    pub episode_id: String,
    pub episode_version: String,
    pub fixture_ref: String,
    pub evaluator_set_ref: String,
    pub components: BTreeMap<ConversationComponentKindV1, ComponentReceiptRefV1>,
    pub commitment: String,
}

impl ConversationQualityEpisodeManifestV1 {
    pub fn new(
        episode_id: impl Into<String>,
        episode_version: impl Into<String>,
        fixture_ref: impl Into<String>,
        evaluator_set_ref: impl Into<String>,
        components: Vec<ComponentReceiptRefV1>,
    ) -> Result<Self, ConversationQualityEpisodeErrorV1> {
        let episode_id = canonical_id(episode_id.into())?;
        let episode_version = canonical_key(episode_version.into())?;
        let fixture_ref = canonical_ref(fixture_ref.into())?;
        let evaluator_set_ref = canonical_ref(evaluator_set_ref.into())?;
        let mut component_map = BTreeMap::new();
        for component in components {
            if component_map.insert(component.kind, component).is_some() {
                return Err(ConversationQualityEpisodeErrorV1::DuplicateComponentKind);
            }
        }
        for required in required_component_kinds() {
            if !component_map.contains_key(&required) {
                return Err(ConversationQualityEpisodeErrorV1::MissingRequiredComponent);
            }
        }
        if component_map.len() != 3 {
            return Err(ConversationQualityEpisodeErrorV1::UnexpectedComponentCount);
        }
        let mut manifest = Self {
            episode_id,
            episode_version,
            fixture_ref,
            evaluator_set_ref,
            components: component_map,
            commitment: String::new(),
        };
        manifest.commitment = manifest_commitment(&manifest);
        Ok(manifest)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SemanticFindingV1 {
    Satisfied,
    Violated,
    NotEstablished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatencyBudgetFindingV1 {
    NotEvaluated,
    NotEstablished,
    WithinBudget,
    Exceeded,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyFindingV1 {
    pub latency_ns: Option<u64>,
    pub budget_finding: LatencyBudgetFindingV1,
}

impl LatencyFindingV1 {
    pub fn new(
        latency_ns: Option<u64>,
        budget_finding: LatencyBudgetFindingV1,
    ) -> Result<Self, ConversationQualityEpisodeErrorV1> {
        let finding = Self {
            latency_ns,
            budget_finding,
        };
        validate_latency_finding(&finding)?;
        Ok(finding)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceEpisodeScorecardV1 {
    pub evidence_state: ComponentEvidenceStateV1,
    pub interruption_semantic: SemanticFindingV1,
    pub interruption_latency: LatencyFindingV1,
    pub stop_semantic: SemanticFindingV1,
    pub stop_latency: LatencyFindingV1,
    pub slowdown_semantic: SemanticFindingV1,
    pub slowdown_latency: LatencyFindingV1,
    pub response_latency: LatencyFindingV1,
    pub backchannel_semantic: SemanticFindingV1,
    pub incomplete_trace_count: u64,
    pub runtime_rejection_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionEpisodeScorecardV1 {
    pub evidence_state: ComponentEvidenceStateV1,
    pub first_compliant_turn_delta: Option<u64>,
    pub first_compliant_latency_ns: Option<u64>,
    pub recurrence_before_adaptation: u64,
    pub recurrence_after_adaptation: u64,
    pub unrelated_context_spillover: u64,
    pub ambiguous_applicable_observations: u64,
    pub applicable_observation_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonaEpisodeScorecardV1 {
    pub evidence_state: ComponentEvidenceStateV1,
    pub critical_violation_count: u64,
    pub noncritical_violation_count: u64,
    pub global_band_violation_count: u64,
    pub context_band_violation_count: u64,
    pub material_disagreement_group_count: u64,
    pub unobserved_required_facet_count: u64,
    pub observed_facet_count: u64,
    pub profile_facet_count: u64,
    pub observed_context_count: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpisodeEvidenceCoverageV1 {
    NotEstablished,
    Partial,
    Complete,
    Uncertain,
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationQualityEpisodeReceiptV1 {
    pub schema: String,
    pub episode_id: String,
    pub manifest_commitment: String,
    pub voice_receipt_commitment: String,
    pub correction_receipt_commitment: String,
    pub persona_receipt_commitment: String,
    pub evidence_coverage: EpisodeEvidenceCoverageV1,
    pub voice: VoiceEpisodeScorecardV1,
    pub correction: CorrectionEpisodeScorecardV1,
    pub persona: PersonaEpisodeScorecardV1,
    pub receipt_commitment: String,
}

impl ConversationQualityEpisodeReceiptV1 {
    pub fn new(
        manifest: &ConversationQualityEpisodeManifestV1,
        voice: VoiceEpisodeScorecardV1,
        correction: CorrectionEpisodeScorecardV1,
        persona: PersonaEpisodeScorecardV1,
    ) -> Result<Self, ConversationQualityEpisodeErrorV1> {
        validate_manifest(manifest)?;
        validate_voice_scorecard(&voice)?;
        validate_correction_scorecard(&correction)?;
        validate_persona_scorecard(&persona)?;

        let voice_ref = component_ref(manifest, ConversationComponentKindV1::FullDuplexVoice);
        let correction_ref = component_ref(manifest, ConversationComponentKindV1::Correction);
        let persona_ref = component_ref(manifest, ConversationComponentKindV1::PersonaContinuity);

        if voice.evidence_state != voice_ref.evidence_state
            || correction.evidence_state != correction_ref.evidence_state
            || persona.evidence_state != persona_ref.evidence_state
        {
            return Err(ConversationQualityEpisodeErrorV1::EvidenceStateMismatch);
        }

        let evidence_coverage = derive_coverage([
            voice.evidence_state,
            correction.evidence_state,
            persona.evidence_state,
        ]);
        let mut receipt = Self {
            schema: CONVERSATION_QUALITY_EPISODE_SCHEMA_V1.into(),
            episode_id: manifest.episode_id.clone(),
            manifest_commitment: manifest.commitment.clone(),
            voice_receipt_commitment: voice_ref.receipt_commitment.clone(),
            correction_receipt_commitment: correction_ref.receipt_commitment.clone(),
            persona_receipt_commitment: persona_ref.receipt_commitment.clone(),
            evidence_coverage,
            voice,
            correction,
            persona,
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = episode_receipt_commitment(&receipt);
        Ok(receipt)
    }

    pub fn validate(
        &self,
        manifest: &ConversationQualityEpisodeManifestV1,
    ) -> Result<(), ConversationQualityEpisodeErrorV1> {
        validate_manifest(manifest)?;
        if self.schema != CONVERSATION_QUALITY_EPISODE_SCHEMA_V1 {
            return Err(ConversationQualityEpisodeErrorV1::InvalidReceiptSchema);
        }
        if self.episode_id != manifest.episode_id || self.manifest_commitment != manifest.commitment {
            return Err(ConversationQualityEpisodeErrorV1::ManifestMismatch);
        }

        validate_voice_scorecard(&self.voice)?;
        validate_correction_scorecard(&self.correction)?;
        validate_persona_scorecard(&self.persona)?;

        let expected = Self::new(
            manifest,
            self.voice.clone(),
            self.correction.clone(),
            self.persona.clone(),
        )?;
        if self.voice_receipt_commitment != expected.voice_receipt_commitment
            || self.correction_receipt_commitment != expected.correction_receipt_commitment
            || self.persona_receipt_commitment != expected.persona_receipt_commitment
            || self.evidence_coverage != expected.evidence_coverage
        {
            return Err(ConversationQualityEpisodeErrorV1::DerivedReceiptFieldMismatch);
        }
        if self.receipt_commitment != episode_receipt_commitment(self)
            || self.receipt_commitment != expected.receipt_commitment
        {
            return Err(ConversationQualityEpisodeErrorV1::ReceiptCommitmentMismatch);
        }
        Ok(())
    }
}

fn required_component_kinds() -> [ConversationComponentKindV1; 3] {
    [
        ConversationComponentKindV1::FullDuplexVoice,
        ConversationComponentKindV1::Correction,
        ConversationComponentKindV1::PersonaContinuity,
    ]
}

fn component_ref(
    manifest: &ConversationQualityEpisodeManifestV1,
    kind: ConversationComponentKindV1,
) -> &ComponentReceiptRefV1 {
    manifest
        .components
        .get(&kind)
        .expect("validated manifest contains every required component")
}

fn validate_manifest(
    manifest: &ConversationQualityEpisodeManifestV1,
) -> Result<(), ConversationQualityEpisodeErrorV1> {
    if manifest.components.len() != 3 {
        return Err(ConversationQualityEpisodeErrorV1::UnexpectedComponentCount);
    }
    if canonical_id(manifest.episode_id.clone())? != manifest.episode_id
        || canonical_key(manifest.episode_version.clone())? != manifest.episode_version
        || canonical_ref(manifest.fixture_ref.clone())? != manifest.fixture_ref
        || canonical_ref(manifest.evaluator_set_ref.clone())? != manifest.evaluator_set_ref
    {
        return Err(ConversationQualityEpisodeErrorV1::NonCanonicalManifestField);
    }
    for required in required_component_kinds() {
        let component = manifest
            .components
            .get(&required)
            .ok_or(ConversationQualityEpisodeErrorV1::MissingRequiredComponent)?;
        let canonical_schema = canonical_key(component.receipt_schema.clone())?;
        if component.kind != required
            || canonical_schema != component.receipt_schema
            || !is_git_sha40(&component.source_head)
            || !is_blake3_commitment(&component.receipt_commitment)
        {
            return Err(ConversationQualityEpisodeErrorV1::InvalidComponentReference);
        }
    }
    if manifest.commitment != manifest_commitment(manifest) {
        return Err(ConversationQualityEpisodeErrorV1::ManifestCommitmentMismatch);
    }
    Ok(())
}

fn validate_latency_finding(
    finding: &LatencyFindingV1,
) -> Result<(), ConversationQualityEpisodeErrorV1> {
    let valid = match finding.budget_finding {
        LatencyBudgetFindingV1::NotEvaluated => true,
        LatencyBudgetFindingV1::NotEstablished => finding.latency_ns.is_none(),
        LatencyBudgetFindingV1::WithinBudget | LatencyBudgetFindingV1::Exceeded => {
            finding.latency_ns.is_some()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(ConversationQualityEpisodeErrorV1::InconsistentLatencyEvidence)
    }
}

fn validate_voice_scorecard(
    scorecard: &VoiceEpisodeScorecardV1,
) -> Result<(), ConversationQualityEpisodeErrorV1> {
    for finding in [
        &scorecard.interruption_latency,
        &scorecard.stop_latency,
        &scorecard.slowdown_latency,
        &scorecard.response_latency,
    ] {
        validate_latency_finding(finding)?;
    }
    Ok(())
}

fn validate_correction_scorecard(
    scorecard: &CorrectionEpisodeScorecardV1,
) -> Result<(), ConversationQualityEpisodeErrorV1> {
    if scorecard.first_compliant_turn_delta.is_some() && scorecard.applicable_observation_count == 0 {
        return Err(ConversationQualityEpisodeErrorV1::InconsistentCorrectionEvidence);
    }
    Ok(())
}

fn validate_persona_scorecard(
    scorecard: &PersonaEpisodeScorecardV1,
) -> Result<(), ConversationQualityEpisodeErrorV1> {
    if scorecard.observed_facet_count > scorecard.profile_facet_count {
        return Err(ConversationQualityEpisodeErrorV1::InconsistentPersonaCoverage);
    }
    Ok(())
}

fn derive_coverage(states: [ComponentEvidenceStateV1; 3]) -> EpisodeEvidenceCoverageV1 {
    if states.contains(&ComponentEvidenceStateV1::Invalid) {
        EpisodeEvidenceCoverageV1::Invalid
    } else if states.contains(&ComponentEvidenceStateV1::Uncertain) {
        EpisodeEvidenceCoverageV1::Uncertain
    } else if states
        .iter()
        .all(|state| *state == ComponentEvidenceStateV1::NotEstablished)
    {
        EpisodeEvidenceCoverageV1::NotEstablished
    } else if states
        .iter()
        .all(|state| *state == ComponentEvidenceStateV1::Established)
    {
        EpisodeEvidenceCoverageV1::Complete
    } else {
        EpisodeEvidenceCoverageV1::Partial
    }
}

fn manifest_commitment(manifest: &ConversationQualityEpisodeManifestV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-conversation-quality-manifest-v1");
    put_str(&mut hasher, &manifest.episode_id);
    put_str(&mut hasher, &manifest.episode_version);
    put_str(&mut hasher, &manifest.fixture_ref);
    put_str(&mut hasher, &manifest.evaluator_set_ref);
    put_u64(&mut hasher, manifest.components.len() as u64);
    for component in manifest.components.values() {
        put_u8(&mut hasher, component_kind_tag(component.kind));
        put_str(&mut hasher, &component.source_head);
        put_str(&mut hasher, &component.receipt_schema);
        put_str(&mut hasher, &component.receipt_commitment);
        put_u8(&mut hasher, component_evidence_tag(component.evidence_state));
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn episode_receipt_commitment(receipt: &ConversationQualityEpisodeReceiptV1) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, "symthaea-conversation-quality-receipt-v1");
    put_str(&mut hasher, &receipt.schema);
    put_str(&mut hasher, &receipt.episode_id);
    put_str(&mut hasher, &receipt.manifest_commitment);
    put_str(&mut hasher, &receipt.voice_receipt_commitment);
    put_str(&mut hasher, &receipt.correction_receipt_commitment);
    put_str(&mut hasher, &receipt.persona_receipt_commitment);
    put_u8(&mut hasher, coverage_tag(receipt.evidence_coverage));
    hash_voice(&mut hasher, &receipt.voice);
    hash_correction(&mut hasher, &receipt.correction);
    hash_persona(&mut hasher, &receipt.persona);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_voice(hasher: &mut blake3::Hasher, value: &VoiceEpisodeScorecardV1) {
    put_u8(hasher, component_evidence_tag(value.evidence_state));
    put_u8(hasher, semantic_tag(value.interruption_semantic));
    hash_latency(hasher, &value.interruption_latency);
    put_u8(hasher, semantic_tag(value.stop_semantic));
    hash_latency(hasher, &value.stop_latency);
    put_u8(hasher, semantic_tag(value.slowdown_semantic));
    hash_latency(hasher, &value.slowdown_latency);
    hash_latency(hasher, &value.response_latency);
    put_u8(hasher, semantic_tag(value.backchannel_semantic));
    put_u64(hasher, value.incomplete_trace_count);
    put_u64(hasher, value.runtime_rejection_count);
}

fn hash_correction(hasher: &mut blake3::Hasher, value: &CorrectionEpisodeScorecardV1) {
    put_u8(hasher, component_evidence_tag(value.evidence_state));
    put_opt_u64(hasher, value.first_compliant_turn_delta);
    put_opt_u64(hasher, value.first_compliant_latency_ns);
    put_u64(hasher, value.recurrence_before_adaptation);
    put_u64(hasher, value.recurrence_after_adaptation);
    put_u64(hasher, value.unrelated_context_spillover);
    put_u64(hasher, value.ambiguous_applicable_observations);
    put_u64(hasher, value.applicable_observation_count);
}

fn hash_persona(hasher: &mut blake3::Hasher, value: &PersonaEpisodeScorecardV1) {
    put_u8(hasher, component_evidence_tag(value.evidence_state));
    put_u64(hasher, value.critical_violation_count);
    put_u64(hasher, value.noncritical_violation_count);
    put_u64(hasher, value.global_band_violation_count);
    put_u64(hasher, value.context_band_violation_count);
    put_u64(hasher, value.material_disagreement_group_count);
    put_u64(hasher, value.unobserved_required_facet_count);
    put_u64(hasher, value.observed_facet_count);
    put_u64(hasher, value.profile_facet_count);
    put_u64(hasher, value.observed_context_count);
}

fn hash_latency(hasher: &mut blake3::Hasher, value: &LatencyFindingV1) {
    put_opt_u64(hasher, value.latency_ns);
    put_u8(hasher, budget_tag(value.budget_finding));
}

fn canonical_id(value: String) -> Result<String, ConversationQualityEpisodeErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 256 {
        return Err(ConversationQualityEpisodeErrorV1::InvalidIdentity);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(ConversationQualityEpisodeErrorV1::InvalidIdentity);
    }
    Ok(value)
}

fn canonical_key(value: String) -> Result<String, ConversationQualityEpisodeErrorV1> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() || value.len() > 256 {
        return Err(ConversationQualityEpisodeErrorV1::InvalidKey);
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '/'))
    {
        return Err(ConversationQualityEpisodeErrorV1::InvalidKey);
    }
    Ok(value)
}

fn canonical_ref(value: String) -> Result<String, ConversationQualityEpisodeErrorV1> {
    let value = value.trim().to_owned();
    if value.is_empty() || value.len() > 1024 {
        return Err(ConversationQualityEpisodeErrorV1::InvalidReference);
    }
    Ok(value)
}

fn is_git_sha40(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn is_blake3_commitment(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn put_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn put_opt_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            put_u8(hasher, 1);
            put_u64(hasher, value);
        }
        None => put_u8(hasher, 0),
    }
}

const fn component_kind_tag(value: ConversationComponentKindV1) -> u8 {
    match value {
        ConversationComponentKindV1::FullDuplexVoice => 1,
        ConversationComponentKindV1::Correction => 2,
        ConversationComponentKindV1::PersonaContinuity => 3,
    }
}

const fn component_evidence_tag(value: ComponentEvidenceStateV1) -> u8 {
    match value {
        ComponentEvidenceStateV1::NotEstablished => 1,
        ComponentEvidenceStateV1::Partial => 2,
        ComponentEvidenceStateV1::Established => 3,
        ComponentEvidenceStateV1::Uncertain => 4,
        ComponentEvidenceStateV1::Invalid => 5,
    }
}

const fn semantic_tag(value: SemanticFindingV1) -> u8 {
    match value {
        SemanticFindingV1::Satisfied => 1,
        SemanticFindingV1::Violated => 2,
        SemanticFindingV1::NotEstablished => 3,
    }
}

const fn budget_tag(value: LatencyBudgetFindingV1) -> u8 {
    match value {
        LatencyBudgetFindingV1::NotEvaluated => 1,
        LatencyBudgetFindingV1::NotEstablished => 2,
        LatencyBudgetFindingV1::WithinBudget => 3,
        LatencyBudgetFindingV1::Exceeded => 4,
    }
}

const fn coverage_tag(value: EpisodeEvidenceCoverageV1) -> u8 {
    match value {
        EpisodeEvidenceCoverageV1::NotEstablished => 1,
        EpisodeEvidenceCoverageV1::Partial => 2,
        EpisodeEvidenceCoverageV1::Complete => 3,
        EpisodeEvidenceCoverageV1::Uncertain => 4,
        EpisodeEvidenceCoverageV1::Invalid => 5,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversationQualityEpisodeErrorV1 {
    InvalidIdentity,
    InvalidKey,
    InvalidReference,
    InvalidSourceHead,
    InvalidReceiptCommitment,
    InvalidReceiptSchema,
    DuplicateComponentKind,
    MissingRequiredComponent,
    UnexpectedComponentCount,
    InvalidComponentReference,
    NonCanonicalManifestField,
    ManifestCommitmentMismatch,
    ManifestMismatch,
    EvidenceStateMismatch,
    DerivedReceiptFieldMismatch,
    InconsistentLatencyEvidence,
    InconsistentCorrectionEvidence,
    InconsistentPersonaCoverage,
    ReceiptCommitmentMismatch,
}
