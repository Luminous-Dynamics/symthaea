// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complete prompt-inventory verification for preference influence decisions.
//!
//! Positive evidence links prove that a particular source span was exposed.
//! They cannot prove that an item was not exposed. This module adds a complete
//! source-to-prompt inventory, bound to exact context and provider-request bytes,
//! so Ignore / Support / Dominate decisions can be checked at the actual model
//! invocation boundary.
//!
//! A self-reported list or a missing retrieval row is not a negative exposure
//! proof. The injected verifier must authenticate the prompt assembler's
//! completeness receipt, each source/transformation mapping, the #13153
//! preference-selection receipt, and the provider request capture.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

fn put_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn put_text(hasher: &mut Sha256, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

fn put_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn digest_hex(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn digest_bytes(bytes: &[u8]) -> String {
    digest_hex(Sha256::digest(bytes))
}

fn lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_len
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreferenceInfluenceModeV1 {
    Ignore,
    Support,
    Dominate,
}

impl PreferenceInfluenceModeV1 {
    fn code(self) -> u8 {
        match self {
            Self::Ignore => 1,
            Self::Support => 2,
            Self::Dominate => 3,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreferenceInfluenceSelectionV1 {
    pub selection_id: String,
    pub run_id: String,
    pub episode_id: String,
    pub action_sequence: u64,
    pub subject_id: String,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub valid_at_ms: i64,
    pub valid_from_ms: i64,
    pub valid_until_ms: Option<i64>,
    pub known_at_ms: i64,
    pub known_from_ms: i64,
    pub known_until_ms: Option<i64>,
    pub decision_policy_sha256: String,
    pub selection_authority_receipt_sha256: String,
    pub mode: PreferenceInfluenceModeV1,
    /// Canonically sorted IDs of the preferences whose influence is being decided.
    pub preference_ids: Vec<String>,
    /// Required only for Dominate; it must be issued by an independent authority.
    pub dominant_preference_authority_receipt_sha256: Option<String>,
    pub canonical_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreferenceExposureRejectV1 {
    InvalidSelection,
    SelectionIntegrityMismatch,
    InvalidManifest,
    ManifestIntegrityMismatch,
    SelectionManifestMismatch,
    InvalidDigest,
    InvalidIdentity,
    InvalidSnapshot,
    InvalidTimestampOrder,
    InvalidPreferenceIdSet,
    InvalidSourceMapping,
    DuplicateSourceMapping,
    DuplicateSpan,
    NonCanonicalSpanOrder,
    OverlappingMemorySpans,
    SpanOutOfBounds,
    SpanNotUtf8Boundary,
    SpanDigestMismatch,
    ModelContextDigestMismatch,
    ProviderRequestDigestMismatch,
    SelectionVerifierRejected,
    DominanceAuthorityMissing,
    DominanceAuthorityRejected,
    SourceMappingRejected,
    PromptInventoryIncomplete,
    ProviderCaptureRejected,
    IgnoredPreferenceWasExposed,
    RequiredPreferenceNotExposed,
    SerializationFailure,
}

impl PreferenceExposureRejectV1 {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidSelection => "invalid_selection",
            Self::SelectionIntegrityMismatch => "selection_integrity_mismatch",
            Self::InvalidManifest => "invalid_manifest",
            Self::ManifestIntegrityMismatch => "manifest_integrity_mismatch",
            Self::SelectionManifestMismatch => "selection_manifest_mismatch",
            Self::InvalidDigest => "invalid_digest",
            Self::InvalidIdentity => "invalid_identity",
            Self::InvalidSnapshot => "invalid_snapshot",
            Self::InvalidTimestampOrder => "invalid_timestamp_order",
            Self::InvalidPreferenceIdSet => "invalid_preference_id_set",
            Self::InvalidSourceMapping => "invalid_source_mapping",
            Self::DuplicateSourceMapping => "duplicate_source_mapping",
            Self::DuplicateSpan => "duplicate_span",
            Self::NonCanonicalSpanOrder => "non_canonical_span_order",
            Self::OverlappingMemorySpans => "overlapping_memory_spans",
            Self::SpanOutOfBounds => "span_out_of_bounds",
            Self::SpanNotUtf8Boundary => "span_not_utf8_boundary",
            Self::SpanDigestMismatch => "span_digest_mismatch",
            Self::ModelContextDigestMismatch => "model_context_digest_mismatch",
            Self::ProviderRequestDigestMismatch => "provider_request_digest_mismatch",
            Self::SelectionVerifierRejected => "selection_verifier_rejected",
            Self::DominanceAuthorityMissing => "dominance_authority_missing",
            Self::DominanceAuthorityRejected => "dominance_authority_rejected",
            Self::SourceMappingRejected => "source_mapping_rejected",
            Self::PromptInventoryIncomplete => "prompt_inventory_incomplete",
            Self::ProviderCaptureRejected => "provider_capture_rejected",
            Self::IgnoredPreferenceWasExposed => "ignored_preference_was_exposed",
            Self::RequiredPreferenceNotExposed => "required_preference_not_exposed",
            Self::SerializationFailure => "serialization_failure",
        }
    }
}

fn selection_digest(selection: &PreferenceInfluenceSelectionV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:preference-influence-selection:v1\0");
    for value in [
        selection.selection_id.as_str(),
        selection.run_id.as_str(),
        selection.episode_id.as_str(),
        selection.subject_id.as_str(),
        selection.scope_id.as_str(),
        selection.purpose_id.as_str(),
        selection.audience_id.as_str(),
        selection.decision_policy_sha256.as_str(),
        selection.selection_authority_receipt_sha256.as_str(),
    ] {
        put_text(&mut hasher, value);
    }
    put_u64(&mut hasher, selection.action_sequence);
    hasher.update(selection.valid_at_ms.to_be_bytes());
    hasher.update(selection.valid_from_ms.to_be_bytes());
    match selection.valid_until_ms {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update(selection.known_at_ms.to_be_bytes());
    hasher.update(selection.known_from_ms.to_be_bytes());
    match selection.known_until_ms {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update([selection.mode.code()]);
    put_u64(&mut hasher, selection.preference_ids.len() as u64);
    for preference_id in &selection.preference_ids {
        put_text(&mut hasher, preference_id);
    }
    match selection.dominant_preference_authority_receipt_sha256.as_deref() {
        Some(value) => {
            hasher.update([1]);
            put_text(&mut hasher, value);
        }
        None => hasher.update([0]),
    }
    digest_hex(hasher)
}

impl PreferenceInfluenceSelectionV1 {
    pub fn seal_integrity(
        mut self,
    ) -> Result<Self, PreferenceExposureRejectV1> {
        if !nonempty(&self.selection_id, 128)
            || !nonempty(&self.run_id, 128)
            || !nonempty(&self.episode_id, 128)
            || !nonempty(&self.subject_id, 256)
            || !nonempty(&self.scope_id, 256)
            || !nonempty(&self.purpose_id, 128)
            || !nonempty(&self.audience_id, 256)
            || !lower_sha256(&self.decision_policy_sha256)
            || !lower_sha256(&self.selection_authority_receipt_sha256)
            || self.preference_ids.is_empty()
            || self.valid_from_ms > self.valid_at_ms
            || self.valid_until_ms.is_some_and(|end| self.valid_at_ms >= end)
            || self.known_from_ms > self.known_at_ms
            || self.known_until_ms.is_some_and(|end| self.known_at_ms >= end)
        {
            return Err(PreferenceExposureRejectV1::InvalidSelection);
        }
        if self.preference_ids.iter().any(|id| !nonempty(id, 256))
            || self.preference_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(PreferenceExposureRejectV1::InvalidPreferenceIdSet);
        }
        match (self.mode, self.dominant_preference_authority_receipt_sha256.as_deref()) {
            (PreferenceInfluenceModeV1::Dominate, Some(value)) if lower_sha256(value) => {}
            (PreferenceInfluenceModeV1::Dominate, _) => {
                return Err(PreferenceExposureRejectV1::DominanceAuthorityMissing);
            }
            (_, None) => {}
            (_, Some(_)) => return Err(PreferenceExposureRejectV1::InvalidSelection),
        }
        self.canonical_sha256.clear();
        self.canonical_sha256 = selection_digest(&self);
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        lower_sha256(&self.canonical_sha256)
            && self.canonical_sha256 == selection_digest(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextMemorySourceV1 {
    pub memory_item_id: String,
    /// Empty means ordinary memory, not a selected preference.
    pub preference_ids: Vec<String>,
    pub source_receipt_sha256: String,
    pub lineage_receipt_sha256: String,
    pub transformation_chain_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextMemorySpanV1 {
    pub span_id: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub representation_sha256: String,
    /// A composed summary can map to multiple source records/preferences.
    pub sources: Vec<ContextMemorySourceV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextExposureManifestV1 {
    pub manifest_id: String,
    pub run_id: String,
    pub episode_id: String,
    pub action_sequence: u64,
    pub selection_id: String,
    pub selection_sha256: String,
    pub subject_id: String,
    pub scope_id: String,
    pub purpose_id: String,
    pub audience_id: String,
    pub valid_at_ms: i64,
    pub known_at_ms: i64,
    pub model_invocation_id: String,
    pub model_context_sha256: String,
    pub provider_request_sha256: String,
    pub prompt_assembler_build_sha256: String,
    pub provider_adapter_build_sha256: String,
    pub prompt_assembly_receipt_sha256: String,
    pub provider_request_capture_receipt_sha256: String,
    pub exposure_observed_at_ms: i64,
    /// Complete, canonical inventory of every memory-origin span in the context.
    pub memory_spans: Vec<ContextMemorySpanV1>,
    pub canonical_sha256: String,
}

fn manifest_digest(manifest: &ContextExposureManifestV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:context-exposure-manifest:v1\0");
    for value in [
        manifest.manifest_id.as_str(),
        manifest.run_id.as_str(),
        manifest.episode_id.as_str(),
        manifest.selection_id.as_str(),
        manifest.selection_sha256.as_str(),
        manifest.subject_id.as_str(),
        manifest.scope_id.as_str(),
        manifest.purpose_id.as_str(),
        manifest.audience_id.as_str(),
        manifest.model_invocation_id.as_str(),
        manifest.model_context_sha256.as_str(),
        manifest.provider_request_sha256.as_str(),
        manifest.prompt_assembler_build_sha256.as_str(),
        manifest.provider_adapter_build_sha256.as_str(),
        manifest.prompt_assembly_receipt_sha256.as_str(),
        manifest.provider_request_capture_receipt_sha256.as_str(),
    ] {
        put_text(&mut hasher, value);
    }
    put_u64(&mut hasher, manifest.action_sequence);
    for value in [
        manifest.valid_at_ms,
        manifest.known_at_ms,
        manifest.exposure_observed_at_ms,
    ] {
        hasher.update(value.to_be_bytes());
    }
    put_u64(&mut hasher, manifest.memory_spans.len() as u64);
    for span in &manifest.memory_spans {
        put_text(&mut hasher, &span.span_id);
        put_u64(&mut hasher, span.byte_start as u64);
        put_u64(&mut hasher, span.byte_end as u64);
        put_text(&mut hasher, &span.representation_sha256);
        put_u64(&mut hasher, span.sources.len() as u64);
        for source in &span.sources {
            put_text(&mut hasher, &source.memory_item_id);
            put_u64(&mut hasher, source.preference_ids.len() as u64);
            for preference_id in &source.preference_ids {
                put_text(&mut hasher, preference_id);
            }
            put_text(&mut hasher, &source.source_receipt_sha256);
            put_text(&mut hasher, &source.lineage_receipt_sha256);
            put_text(&mut hasher, &source.transformation_chain_sha256);
        }
    }
    digest_hex(hasher)
}

impl ContextExposureManifestV1 {
    pub fn seal_integrity(
        mut self,
    ) -> Result<Self, PreferenceExposureRejectV1> {
        validate_manifest_fields(&self)?;
        self.canonical_sha256.clear();
        self.canonical_sha256 = manifest_digest(&self);
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        self.canonical_sha256 == manifest_digest(self)
            && lower_sha256(&self.canonical_sha256)
    }
}

fn validate_manifest_fields(
    manifest: &ContextExposureManifestV1,
) -> Result<(), PreferenceExposureRejectV1> {
    if !nonempty(&manifest.manifest_id, 128)
        || !nonempty(&manifest.run_id, 128)
        || !nonempty(&manifest.episode_id, 128)
        || !nonempty(&manifest.selection_id, 128)
        || !nonempty(&manifest.subject_id, 256)
        || !nonempty(&manifest.scope_id, 256)
        || !nonempty(&manifest.purpose_id, 128)
        || !nonempty(&manifest.audience_id, 256)
        || !nonempty(&manifest.model_invocation_id, 256)
    {
        return Err(PreferenceExposureRejectV1::InvalidIdentity);
    }
    for value in [
        manifest.selection_sha256.as_str(),
        manifest.model_context_sha256.as_str(),
        manifest.provider_request_sha256.as_str(),
        manifest.prompt_assembler_build_sha256.as_str(),
        manifest.provider_adapter_build_sha256.as_str(),
        manifest.prompt_assembly_receipt_sha256.as_str(),
        manifest.provider_request_capture_receipt_sha256.as_str(),
    ] {
        if !lower_sha256(value) {
            return Err(PreferenceExposureRejectV1::InvalidDigest);
        }
    }
    if manifest.valid_at_ms < 0
        || manifest.known_at_ms < 0
        || manifest.exposure_observed_at_ms < manifest.known_at_ms
    {
        return Err(PreferenceExposureRejectV1::InvalidTimestampOrder);
    }
    Ok(())
}

/// The verifier must authenticate the #13153 selection, the prompt assembler's
/// *complete* inventory, each source/transformation mapping, and the bytes
/// actually submitted by the provider adapter. Implementations must reject an
/// unknown/uninstrumented memory injection instead of treating the inventory
/// as complete.
pub trait ContextExposureVerifierV1 {
    fn verify_preference_selection(
        &self,
        selection: &PreferenceInfluenceSelectionV1,
    ) -> Result<(), PreferenceExposureRejectV1>;

    fn verify_dominance_authority(
        &self,
        selection: &PreferenceInfluenceSelectionV1,
    ) -> Result<(), PreferenceExposureRejectV1>;

    fn verify_memory_source(
        &self,
        source: &ContextMemorySourceV1,
        selection: &PreferenceInfluenceSelectionV1,
    ) -> Result<(), PreferenceExposureRejectV1>;

    fn verify_complete_prompt_inventory(
        &self,
        manifest: &ContextExposureManifestV1,
        model_context_bytes: &[u8],
    ) -> Result<(), PreferenceExposureRejectV1>;

    fn verify_provider_request_capture(
        &self,
        manifest: &ContextExposureManifestV1,
        provider_request_bytes: &[u8],
        model_context_bytes: &[u8],
    ) -> Result<(), PreferenceExposureRejectV1>;
}

/// Private in-process capability. A serialized manifest or recomputed hash
/// cannot recreate this type; the trusted verifier must be run again.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedPreferenceExposureV1 {
    selection_id: String,
    manifest_sha256: String,
    mode: PreferenceInfluenceModeV1,
    exposed_preference_ids: BTreeSet<String>,
}

impl VerifiedPreferenceExposureV1 {
    pub fn selection_id(&self) -> &str {
        &self.selection_id
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    pub fn mode(&self) -> PreferenceInfluenceModeV1 {
        self.mode
    }

    pub fn preference_was_exposed(&self, preference_id: &str) -> bool {
        self.exposed_preference_ids.contains(preference_id)
    }

    pub fn exposed_preference_ids(&self) -> impl Iterator<Item = &str> {
        self.exposed_preference_ids.iter().map(String::as_str)
    }
}

pub fn verify_preference_exposure_v1<V: ContextExposureVerifierV1>(
    selection: &PreferenceInfluenceSelectionV1,
    manifest: &ContextExposureManifestV1,
    model_context_bytes: &[u8],
    provider_request_bytes: &[u8],
    verifier: &V,
) -> Result<VerifiedPreferenceExposureV1, PreferenceExposureRejectV1> {
    if !selection.verify_integrity() {
        return Err(PreferenceExposureRejectV1::SelectionIntegrityMismatch);
    }
    validate_manifest_fields(manifest)?;
    if !manifest.verify_integrity() {
        return Err(PreferenceExposureRejectV1::ManifestIntegrityMismatch);
    }
    if manifest.selection_id != selection.selection_id
        || manifest.selection_sha256 != selection.canonical_sha256
        || manifest.run_id != selection.run_id
        || manifest.episode_id != selection.episode_id
        || manifest.action_sequence != selection.action_sequence
        || manifest.subject_id != selection.subject_id
        || manifest.scope_id != selection.scope_id
        || manifest.purpose_id != selection.purpose_id
        || manifest.audience_id != selection.audience_id
        || manifest.valid_at_ms != selection.valid_at_ms
        || manifest.known_at_ms != selection.known_at_ms
    {
        return Err(PreferenceExposureRejectV1::SelectionManifestMismatch);
    }
    if digest_bytes(model_context_bytes) != manifest.model_context_sha256 {
        return Err(PreferenceExposureRejectV1::ModelContextDigestMismatch);
    }
    if digest_bytes(provider_request_bytes) != manifest.provider_request_sha256 {
        return Err(PreferenceExposureRejectV1::ProviderRequestDigestMismatch);
    }

    verifier
        .verify_preference_selection(selection)
        .map_err(|_| PreferenceExposureRejectV1::SelectionVerifierRejected)?;
    if selection.mode == PreferenceInfluenceModeV1::Dominate {
        if selection.dominant_preference_authority_receipt_sha256.is_none() {
            return Err(PreferenceExposureRejectV1::DominanceAuthorityMissing);
        }
        verifier
            .verify_dominance_authority(selection)
            .map_err(|_| PreferenceExposureRejectV1::DominanceAuthorityRejected)?;
    }

    let context_text = std::str::from_utf8(model_context_bytes)
        .map_err(|_| PreferenceExposureRejectV1::SpanNotUtf8Boundary)?;
    let mut previous_span: Option<(usize, usize, &str)> = None;
    let mut span_ids = BTreeSet::new();
    let mut source_keys = BTreeSet::new();
    let mut exposed_preference_ids = BTreeSet::new();

    for span in &manifest.memory_spans {
        if !nonempty(&span.span_id, 128)
            || !span_ids.insert(span.span_id.as_str())
        {
            return Err(PreferenceExposureRejectV1::DuplicateSpan);
        }
        if span.byte_start >= span.byte_end || span.sources.is_empty() {
            return Err(PreferenceExposureRejectV1::InvalidSourceMapping);
        }
        if span.byte_end > model_context_bytes.len() {
            return Err(PreferenceExposureRejectV1::SpanOutOfBounds);
        }
        if !context_text.is_char_boundary(span.byte_start)
            || !context_text.is_char_boundary(span.byte_end)
        {
            return Err(PreferenceExposureRejectV1::SpanNotUtf8Boundary);
        }
        let current = (span.byte_start, span.byte_end, span.span_id.as_str());
        if previous_span.is_some_and(|prior| prior >= current) {
            return Err(PreferenceExposureRejectV1::NonCanonicalSpanOrder);
        }
        if previous_span.is_some_and(|prior| prior.1 > span.byte_start) {
            return Err(PreferenceExposureRejectV1::OverlappingMemorySpans);
        }
        previous_span = Some(current);
        if digest_bytes(&model_context_bytes[span.byte_start..span.byte_end])
            != span.representation_sha256
            || !lower_sha256(&span.representation_sha256)
        {
            return Err(PreferenceExposureRejectV1::SpanDigestMismatch);
        }

        let mut prior_source_id: Option<&str> = None;
        for source in &span.sources {
            if !nonempty(&source.memory_item_id, 256)
                || !lower_sha256(&source.source_receipt_sha256)
                || !lower_sha256(&source.lineage_receipt_sha256)
                || !lower_sha256(&source.transformation_chain_sha256)
            {
                return Err(PreferenceExposureRejectV1::InvalidSourceMapping);
            }
            if prior_source_id.is_some_and(|prior| prior >= source.memory_item_id.as_str()) {
                return Err(PreferenceExposureRejectV1::DuplicateSourceMapping);
            }
            prior_source_id = Some(source.memory_item_id.as_str());
            if source.preference_ids.iter().any(|id| !nonempty(id, 256))
                || source.preference_ids.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(PreferenceExposureRejectV1::InvalidSourceMapping);
            }
            if !source_keys.insert(source.memory_item_id.as_str()) {
                return Err(PreferenceExposureRejectV1::DuplicateSourceMapping);
            }
            for preference_id in &source.preference_ids {
                exposed_preference_ids.insert(preference_id.clone());
            }
            verifier
                .verify_memory_source(source, selection)
                .map_err(|_| PreferenceExposureRejectV1::SourceMappingRejected)?;
        }
    }

    verifier
        .verify_complete_prompt_inventory(manifest, model_context_bytes)
        .map_err(|_| PreferenceExposureRejectV1::PromptInventoryIncomplete)?;
    verifier
        .verify_provider_request_capture(manifest, provider_request_bytes, model_context_bytes)
        .map_err(|_| PreferenceExposureRejectV1::ProviderCaptureRejected)?;

    let selected: BTreeSet<&str> = selection.preference_ids.iter().map(String::as_str).collect();
    let exposed_selected = selected
        .iter()
        .filter(|preference_id| exposed_preference_ids.contains(**preference_id))
        .count();
    match selection.mode {
        PreferenceInfluenceModeV1::Ignore => {
            if exposed_selected != 0 {
                return Err(PreferenceExposureRejectV1::IgnoredPreferenceWasExposed);
            }
        }
        PreferenceInfluenceModeV1::Support | PreferenceInfluenceModeV1::Dominate => {
            if exposed_selected != selected.len() {
                return Err(PreferenceExposureRejectV1::RequiredPreferenceNotExposed);
            }
        }
    }

    Ok(VerifiedPreferenceExposureV1 {
        selection_id: selection.selection_id.clone(),
        manifest_sha256: manifest.canonical_sha256.clone(),
        mode: selection.mode,
        exposed_preference_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Default)]
    struct FixtureVerifier {
        inventory_complete: bool,
        provider_capture_valid: bool,
        dominance_authorized: bool,
    }

    impl ContextExposureVerifierV1 for FixtureVerifier {
        fn verify_preference_selection(
            &self,
            selection: &PreferenceInfluenceSelectionV1,
        ) -> Result<(), PreferenceExposureRejectV1> {
            if selection.selection_authority_receipt_sha256 == digest_bytes(b"selection-authority") {
                Ok(())
            } else {
                Err(PreferenceExposureRejectV1::SelectionVerifierRejected)
            }
        }

        fn verify_dominance_authority(
            &self,
            selection: &PreferenceInfluenceSelectionV1,
        ) -> Result<(), PreferenceExposureRejectV1> {
            if self.dominance_authorized
                && selection.dominant_preference_authority_receipt_sha256
                    == Some(digest_bytes(b"dominance-authority"))
            {
                Ok(())
            } else {
                Err(PreferenceExposureRejectV1::DominanceAuthorityRejected)
            }
        }

        fn verify_memory_source(
            &self,
            source: &ContextMemorySourceV1,
            _selection: &PreferenceInfluenceSelectionV1,
        ) -> Result<(), PreferenceExposureRejectV1> {
            if source.source_receipt_sha256 == digest_bytes(b"source")
                && source.lineage_receipt_sha256 == digest_bytes(b"lineage")
                && source.transformation_chain_sha256 == digest_bytes(b"transform")
            {
                Ok(())
            } else {
                Err(PreferenceExposureRejectV1::SourceMappingRejected)
            }
        }

        fn verify_complete_prompt_inventory(
            &self,
            manifest: &ContextExposureManifestV1,
            _model_context_bytes: &[u8],
        ) -> Result<(), PreferenceExposureRejectV1> {
            if self.inventory_complete
                && manifest.prompt_assembly_receipt_sha256 == digest_bytes(b"prompt-assembly")
            {
                Ok(())
            } else {
                Err(PreferenceExposureRejectV1::PromptInventoryIncomplete)
            }
        }

        fn verify_provider_request_capture(
            &self,
            manifest: &ContextExposureManifestV1,
            _provider_request_bytes: &[u8],
            _model_context_bytes: &[u8],
        ) -> Result<(), PreferenceExposureRejectV1> {
            if self.provider_capture_valid
                && manifest.provider_request_capture_receipt_sha256 == digest_bytes(b"provider-capture")
            {
                Ok(())
            } else {
                Err(PreferenceExposureRejectV1::ProviderCaptureRejected)
            }
        }
    }

    fn digest(bytes: &[u8]) -> String {
        digest_bytes(bytes)
    }

    fn selection(mode: PreferenceInfluenceModeV1) -> PreferenceInfluenceSelectionV1 {
        PreferenceInfluenceSelectionV1 {
            selection_id: "selection-1".into(),
            run_id: "run-1".into(),
            episode_id: "episode-1".into(),
            action_sequence: 4,
            subject_id: "user:alice".into(),
            scope_id: "workspace:alpha".into(),
            purpose_id: "assist:calendar".into(),
            audience_id: "user:alice".into(),
            valid_at_ms: 100,
            valid_from_ms: 0,
            valid_until_ms: Some(1_000),
            known_at_ms: 100,
            known_from_ms: 0,
            known_until_ms: Some(1_000),
            decision_policy_sha256: digest(b"policy"),
            selection_authority_receipt_sha256: digest(b"selection-authority"),
            mode,
            preference_ids: vec!["pref:outreach".into()],
            dominant_preference_authority_receipt_sha256: (mode == PreferenceInfluenceModeV1::Dominate)
                .then(|| digest(b"dominance-authority")),
            canonical_sha256: String::new(),
        }
        .seal_integrity()
        .unwrap()
    }

    fn source(preference_ids: Vec<String>, item: &str) -> ContextMemorySourceV1 {
        ContextMemorySourceV1 {
            memory_item_id: item.into(),
            preference_ids,
            source_receipt_sha256: digest(b"source"),
            lineage_receipt_sha256: digest(b"lineage"),
            transformation_chain_sha256: digest(b"transform"),
        }
    }

    fn span(
        context: &[u8],
        span_id: &str,
        start: usize,
        end: usize,
        sources: Vec<ContextMemorySourceV1>,
    ) -> ContextMemorySpanV1 {
        ContextMemorySpanV1 {
            span_id: span_id.into(),
            byte_start: start,
            byte_end: end,
            representation_sha256: digest(&context[start..end]),
            sources,
        }
    }

    fn manifest(
        selection: &PreferenceInfluenceSelectionV1,
        context: &[u8],
        request: &[u8],
        spans: Vec<ContextMemorySpanV1>,
    ) -> ContextExposureManifestV1 {
        ContextExposureManifestV1 {
            manifest_id: "manifest-1".into(),
            run_id: selection.run_id.clone(),
            episode_id: selection.episode_id.clone(),
            action_sequence: selection.action_sequence,
            selection_id: selection.selection_id.clone(),
            selection_sha256: selection.canonical_sha256.clone(),
            subject_id: selection.subject_id.clone(),
            scope_id: selection.scope_id.clone(),
            purpose_id: selection.purpose_id.clone(),
            audience_id: selection.audience_id.clone(),
            valid_at_ms: selection.valid_at_ms,
            known_at_ms: selection.known_at_ms,
            model_invocation_id: "provider-call-1".into(),
            model_context_sha256: digest(context),
            provider_request_sha256: digest(request),
            prompt_assembler_build_sha256: digest(b"assembler-build"),
            provider_adapter_build_sha256: digest(b"provider-adapter-build"),
            prompt_assembly_receipt_sha256: digest(b"prompt-assembly"),
            provider_request_capture_receipt_sha256: digest(b"provider-capture"),
            exposure_observed_at_ms: 110,
            memory_spans: spans,
            canonical_sha256: String::new(),
        }
        .seal_integrity()
        .unwrap()
    }

    fn trusted_verifier() -> FixtureVerifier {
        FixtureVerifier {
            inventory_complete: true,
            provider_capture_valid: true,
            dominance_authorized: true,
        }
    }

    #[test]
    fn ignore_is_verified_only_after_complete_prompt_inventory_and_provider_capture() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"request text plus unrelated memory excerpt";
        let request = b"{\"model\":\"fixture\",\"input\":\"request text plus unrelated memory excerpt\"}";
        let unrelated = b"unrelated memory excerpt";
        let start = context
            .windows(unrelated.len())
            .position(|window| window == unrelated)
            .unwrap();
        let manifest = manifest(
            &selection,
            context,
            request,
            vec![span(context, "span-1", start, start + unrelated.len(), vec![source(vec![], "mem:other")])],
        );
        let verified = verify_preference_exposure_v1(
            &selection,
            &manifest,
            context,
            request,
            &trusted_verifier(),
        )
        .unwrap();
        assert_eq!(verified.mode(), PreferenceInfluenceModeV1::Ignore);
        assert!(!verified.preference_was_exposed("pref:outreach"));
    }

    #[test]
    fn ignore_fails_when_selected_preference_is_in_a_composed_memory_span() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"summary: user prefers no unsolicited outreach";
        let request = b"{\"input\":\"summary: user prefers no unsolicited outreach\"}";
        let manifest = manifest(
            &selection,
            context,
            request,
            vec![span(
                context,
                "span-1",
                0,
                context.len(),
                vec![source(vec!["pref:outreach".into()], "mem:summary-1")],
            )],
        );
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &trusted_verifier()),
            Err(PreferenceExposureRejectV1::IgnoredPreferenceWasExposed)
        );
    }

    #[test]
    fn missing_or_incomplete_prompt_inventory_is_not_negative_exposure_evidence() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"plain request";
        let request = b"{\"input\":\"plain request\"}";
        let manifest = manifest(&selection, context, request, vec![]);
        let verifier = FixtureVerifier {
            inventory_complete: false,
            provider_capture_valid: true,
            dominance_authorized: false,
        };
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &verifier),
            Err(PreferenceExposureRejectV1::PromptInventoryIncomplete)
        );
    }

    #[test]
    fn support_requires_each_selected_preference_to_be_actually_exposed() {
        let selection = selection(PreferenceInfluenceModeV1::Support);
        let context = b"the user prefers no unsolicited outreach";
        let request = b"{\"input\":\"the user prefers no unsolicited outreach\"}";
        let manifest = manifest(
            &selection,
            context,
            request,
            vec![span(
                context,
                "span-1",
                0,
                context.len(),
                vec![source(vec!["pref:outreach".into()], "mem:pref-1")],
            )],
        );
        let verified = verify_preference_exposure_v1(
            &selection,
            &manifest,
            context,
            request,
            &trusted_verifier(),
        )
        .unwrap();
        assert!(verified.preference_was_exposed("pref:outreach"));
    }

    #[test]
    fn support_fails_when_selected_preference_was_not_exposed() {
        let selection = selection(PreferenceInfluenceModeV1::Support);
        let context = b"plain request";
        let request = b"{\"input\":\"plain request\"}";
        let manifest = manifest(&selection, context, request, vec![]);
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &trusted_verifier()),
            Err(PreferenceExposureRejectV1::RequiredPreferenceNotExposed)
        );
    }

    #[test]
    fn dominate_requires_independent_authority_and_positive_exposure() {
        let selection = selection(PreferenceInfluenceModeV1::Dominate);
        let context = b"apply the no-outreach constraint";
        let request = b"{\"input\":\"apply the no-outreach constraint\"}";
        let manifest = manifest(
            &selection,
            context,
            request,
            vec![span(
                context,
                "span-1",
                0,
                context.len(),
                vec![source(vec!["pref:outreach".into()], "mem:pref-1")],
            )],
        );
        let verifier = FixtureVerifier {
            inventory_complete: true,
            provider_capture_valid: true,
            dominance_authorized: false,
        };
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &verifier),
            Err(PreferenceExposureRejectV1::DominanceAuthorityRejected)
        );
    }

    #[test]
    fn exact_provider_request_bytes_are_bound_to_the_inventory() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"plain request";
        let request = b"{\"input\":\"plain request\"}";
        let manifest = manifest(&selection, context, request, vec![]);
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, b"{\"input\":\"changed\"}", &trusted_verifier()),
            Err(PreferenceExposureRejectV1::ProviderRequestDigestMismatch)
        );
    }

    #[test]
    fn source_and_transformation_receipts_are_required_for_every_memory_span() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"unrelated memory";
        let request = b"{\"input\":\"unrelated memory\"}";
        let mut bad_source = source(vec![], "mem:other");
        bad_source.transformation_chain_sha256 = digest(b"unverified-transform");
        let manifest = manifest(
            &selection,
            context,
            request,
            vec![span(context, "span-1", 0, context.len(), vec![bad_source])],
        );
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &trusted_verifier()),
            Err(PreferenceExposureRejectV1::SourceMappingRejected)
        );
    }

    #[test]
    fn memory_spans_must_be_canonical_non_overlapping_utf8_ranges() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"first memory second memory";
        let request = b"{\"input\":\"first memory second memory\"}";
        let overlap = vec![
            span(context, "span-1", 0, 14, vec![source(vec![], "mem:a")]),
            span(context, "span-2", 7, context.len(), vec![source(vec![], "mem:b")]),
        ];
        let manifest = manifest(&selection, context, request, overlap);
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &trusted_verifier()),
            Err(PreferenceExposureRejectV1::OverlappingMemorySpans)
        );
    }

    #[test]
    fn manifest_rejects_identity_and_snapshot_rebinding() {
        let selection = selection(PreferenceInfluenceModeV1::Ignore);
        let context = b"plain request";
        let request = b"{\"input\":\"plain request\"}";
        let mut manifest = manifest(&selection, context, request, vec![]);
        manifest.audience_id = "user:bob".into();
        manifest.canonical_sha256 = manifest_digest(&manifest);
        assert_eq!(
            verify_preference_exposure_v1(&selection, &manifest, context, request, &trusted_verifier()),
            Err(PreferenceExposureRejectV1::SelectionManifestMismatch)
        );
    }
}
