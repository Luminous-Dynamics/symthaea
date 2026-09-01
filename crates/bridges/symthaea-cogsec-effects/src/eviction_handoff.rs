// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical state commitments for the legacy working-memory eviction handoff.
//!
//! `ContinuousMind::evicted_items` is externally drained for graduation and
//! persistence, so it is not merely an implementation detail of working memory.
//! This module gives ObserverOnly qualification a deterministic projection of that
//! buffer without assigning it a frozen-K0 mutation class or any authority.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use symthaea_cogsec::Digest32;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_memory::MemorySource;

use crate::{continuous_hv_digest_v1, metadata_digest_v1};

const EVICTION_HANDOFF_ITEM_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_EVICTION_HANDOFF_ITEM/v1";
const EVICTION_HANDOFF_STATE_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_EVICTION_HANDOFF_STATE/v1";

/// Canonical resource identifier for the legacy eviction/persistence handoff.
pub const EVICTION_HANDOFF_RESOURCE_V1: &str = "mind/memory/eviction-handoff";

/// Read-only dependency-neutral view of one legacy `EvictedMemory` record.
///
/// The bridge deliberately does not depend on the root `symthaea` crate, avoiding
/// a dependency cycle. Runtime code may construct this view from the exact fields
/// of an `EvictedMemory` immediately before/after the legacy handoff mutation.
#[derive(Debug, Clone, Copy)]
pub struct EvictionHandoffItemView<'a> {
    /// Exact evicted HDC content.
    pub content: &'a ContinuousHV,
    /// Number of working-memory ticks survived before eviction.
    pub steps_survived: u64,
    /// Legacy memory source classification.
    pub source: MemorySource,
    /// Legacy verification bit. This remains ordinary data, not CogSec authority.
    pub is_verified: bool,
    /// Exact persistence-tagging metadata.
    pub metadata: &'a HashMap<String, String>,
}

/// Canonical identity of one eviction-handoff record.
///
/// This value proves only deterministic record identity. It is not an owner token,
/// permit, provenance assertion, persistence authorization, or `ResourceVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvictionHandoffItemCommitmentV1 {
    digest: Digest32,
}

impl EvictionHandoffItemCommitmentV1 {
    /// Commit to one exact legacy eviction-handoff record.
    pub fn new(item: EvictionHandoffItemView<'_>) -> Self {
        let mut writer = HandoffWriter::with_domain(EVICTION_HANDOFF_ITEM_DOMAIN_V1);
        writer.digest(continuous_hv_digest_v1(item.content));
        writer.u64(item.steps_survived);
        writer.u8(memory_source_code(item.source));
        writer.bool(item.is_verified);
        writer.digest(metadata_digest_v1(item.metadata));
        Self {
            digest: sha256(&writer.finish()),
        }
    }

    /// Exact canonical record digest.
    pub const fn digest(&self) -> Digest32 {
        self.digest
    }
}

/// Canonical ordered state commitment for the complete eviction-handoff buffer.
///
/// Order is part of the resource state because `take_evicted_tagged()` drains the
/// vector in order and downstream persistence derives per-item identifiers from
/// enumeration order. Empty, append, reorder, field-change, and drain-to-empty
/// states therefore have distinct commitments except where the states are exactly
/// identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvictionHandoffStateCommitmentV1 {
    digest: Digest32,
    count: u64,
}

impl EvictionHandoffStateCommitmentV1 {
    /// Commit to an ordered slice of exact handoff records.
    pub fn new(items: &[EvictionHandoffItemView<'_>]) -> Self {
        let mut writer = HandoffWriter::with_domain(EVICTION_HANDOFF_STATE_DOMAIN_V1);
        writer.u64(items.len() as u64);
        for (index, item) in items.iter().copied().enumerate() {
            writer.u64(index as u64);
            writer.digest(EvictionHandoffItemCommitmentV1::new(item).digest());
        }
        Self {
            digest: sha256(&writer.finish()),
            count: items.len() as u64,
        }
    }

    /// Stable protected-resource identifier for this commitment schema.
    pub const fn resource_name(&self) -> &'static str {
        EVICTION_HANDOFF_RESOURCE_V1
    }

    /// Number of records committed by this state root.
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Whether the committed buffer is empty.
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Exact canonical state digest.
    pub const fn digest(&self) -> Digest32 {
        self.digest
    }
}

fn memory_source_code(source: MemorySource) -> u8 {
    match source {
        MemorySource::Internal => 0,
        MemorySource::WebResearch => 1,
        MemorySource::UserInteraction => 2,
        MemorySource::ActionFeedback => 3,
        MemorySource::SemanticEviction => 4,
        MemorySource::Social => 5,
    }
}

fn sha256(bytes: &[u8]) -> Digest32 {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    Digest32(digest)
}

#[derive(Debug)]
struct HandoffWriter {
    bytes: Vec<u8>,
}

impl HandoffWriter {
    fn with_domain(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 1 + 128);
        bytes.extend_from_slice(domain);
        bytes.push(0);
        Self { bytes }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn digest(&mut self, value: Digest32) {
        self.bytes.extend_from_slice(&value.0);
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(a: f32, b: f32) -> ContinuousHV {
        ContinuousHV::from_values(vec![a, b])
    }

    fn item<'a>(
        content: &'a ContinuousHV,
        metadata: &'a HashMap<String, String>,
        steps: u64,
        source: MemorySource,
        verified: bool,
    ) -> EvictionHandoffItemView<'a> {
        EvictionHandoffItemView {
            content,
            steps_survived: steps,
            source,
            is_verified: verified,
            metadata,
        }
    }

    #[test]
    fn empty_handoff_state_is_deterministic() {
        let a = EvictionHandoffStateCommitmentV1::new(&[]);
        let b = EvictionHandoffStateCommitmentV1::new(&[]);
        assert_eq!(a, b);
        assert!(a.is_empty());
        assert_eq!(a.count(), 0);
        assert_eq!(a.resource_name(), EVICTION_HANDOFF_RESOURCE_V1);
    }

    #[test]
    fn handoff_state_binds_order() {
        let a_hv = hv(0.1, 0.2);
        let b_hv = hv(0.3, 0.4);
        let metadata = HashMap::new();
        let a = item(&a_hv, &metadata, 3, MemorySource::Internal, false);
        let b = item(&b_hv, &metadata, 5, MemorySource::WebResearch, true);

        let forward = EvictionHandoffStateCommitmentV1::new(&[a, b]);
        let reverse = EvictionHandoffStateCommitmentV1::new(&[b, a]);
        assert_ne!(forward, reverse);
    }

    #[test]
    fn handoff_item_binds_every_legacy_field() {
        let content = hv(0.1, 0.2);
        let other_content = hv(0.1, 0.3);
        let mut metadata = HashMap::new();
        metadata.insert("topic".to_string(), "alpha".to_string());
        let mut other_metadata = metadata.clone();
        other_metadata.insert("topic".to_string(), "beta".to_string());

        let base = EvictionHandoffItemCommitmentV1::new(item(
            &content,
            &metadata,
            3,
            MemorySource::Internal,
            false,
        ));

        assert_ne!(
            base,
            EvictionHandoffItemCommitmentV1::new(item(
                &other_content,
                &metadata,
                3,
                MemorySource::Internal,
                false,
            ))
        );
        assert_ne!(
            base,
            EvictionHandoffItemCommitmentV1::new(item(
                &content,
                &metadata,
                4,
                MemorySource::Internal,
                false,
            ))
        );
        assert_ne!(
            base,
            EvictionHandoffItemCommitmentV1::new(item(
                &content,
                &metadata,
                3,
                MemorySource::UserInteraction,
                false,
            ))
        );
        assert_ne!(
            base,
            EvictionHandoffItemCommitmentV1::new(item(
                &content,
                &metadata,
                3,
                MemorySource::Internal,
                true,
            ))
        );
        assert_ne!(
            base,
            EvictionHandoffItemCommitmentV1::new(item(
                &content,
                &other_metadata,
                3,
                MemorySource::Internal,
                false,
            ))
        );
    }

    #[test]
    fn drain_to_empty_changes_state_commitment() {
        let content = hv(0.1, 0.2);
        let metadata = HashMap::new();
        let nonempty = EvictionHandoffStateCommitmentV1::new(&[item(
            &content,
            &metadata,
            3,
            MemorySource::Internal,
            false,
        )]);
        let empty = EvictionHandoffStateCommitmentV1::new(&[]);
        assert_ne!(nonempty, empty);
    }
}
