// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical effect commitments shared by CogSec evaluation and shadow observation.
//!
//! This crate sits **outside** the logical reference-monitor core. The monitor
//! accepts an already-established [`Digest32`]; this bridge defines one stable
//! representation for selected Symthaea cognitive effects so proposal/evaluation
//! code and post-legacy observation code do not independently invent hashes.
//!
//! A matching commitment establishes effect identity only. It does not establish
//! authority, factual correctness, provenance, authentication, or owner freshness.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use symthaea_cogsec::Digest32;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_core::hdc::{HdcDimensionality, LiquidHolocell};
use symthaea_memory::MemorySource;

const EFFECT_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_EFFECT/v1";
const HDC_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_HDC/v1";
const METADATA_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_METADATA/v1";
const HOLOCELL_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_HOLOCELL/v1";
const ACTIVE_STATE_DOMAIN_V1: &[u8] = b"SYMTHAEA_COGSEC_ACTIVE_STATE/v1";

const EFFECT_WM_ADMIT: u8 = 1;
const EFFECT_WM_REPLACE: u8 = 2;
const EFFECT_GRADUATION_ENQUEUE: u8 = 3;
const EFFECT_ACTIVE_STATE_REPLACE: u8 = 4;
const EFFECT_GOAL_ACTIVATE: u8 = 5;
const EFFECT_AFFECT_SET: u8 = 6;

/// Canonical v1 effect families needed by the first S0/S1/S2 shadow-runtime tranche.
///
/// Complex values such as HDC vectors, active owner state, and metadata maps enter
/// as independently canonicalized [`Digest32`] commitments produced by this crate.
/// Float-bearing fields are stored as IEEE bit patterns after the exact legacy
/// computation has produced the value to be committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveEffectV1 {
    /// Admit one item into working memory without eviction.
    WorkingMemoryAdmit {
        /// Commitment to the admitted HDC content.
        item: Digest32,
        /// Arrival tick stored in the parallel working-memory tick array.
        arrival_tick: u64,
        /// Legacy source classification stored with the item.
        source: MemorySource,
        /// Legacy verification flag stored with the item. This is not CogSec authority.
        verified: bool,
        /// Commitment to the stored metadata map.
        metadata: Digest32,
        /// Exact insertion index in the post-operation ordering rule.
        insertion_index: u64,
    },
    /// Replace/evict one working-memory item while admitting another.
    ///
    /// This binds the actual parallel-array transition and the legacy eviction
    /// buffer record. The separate graduation queue write remains its own
    /// [`Self::GraduationEnqueue`] effect so admission/replacement never grants
    /// persistence authority by implication.
    WorkingMemoryReplace {
        /// Commitment to the admitted HDC content.
        admitted_item: Digest32,
        /// Arrival tick stored for the admitted item.
        admitted_arrival_tick: u64,
        /// Legacy source stored for the admitted item.
        admitted_source: MemorySource,
        /// Legacy verification flag stored for the admitted item.
        admitted_verified: bool,
        /// Commitment to the admitted item's metadata map.
        admitted_metadata: Digest32,
        /// Exact insertion index for the admitted item.
        admitted_index: u64,
        /// Commitment to the evicted HDC content.
        evicted_item: Digest32,
        /// Arrival tick removed with the evicted item.
        evicted_arrival_tick: u64,
        /// Exact `steps_survived` written to the legacy eviction buffer.
        evicted_steps_survived: u64,
        /// Legacy source stored for the evicted item.
        evicted_source: MemorySource,
        /// Legacy verification flag stored for the evicted item.
        evicted_verified: bool,
        /// Commitment to the evicted item's metadata map.
        evicted_metadata: Digest32,
        /// Exact index removed by the replacement.
        evicted_index: u64,
    },
    /// Enqueue one graduation/persistence candidate.
    GraduationEnqueue {
        /// Commitment to the queued HDC content.
        content: Digest32,
        /// Exact queued label.
        label: String,
        /// Legacy number of steps survived before graduation.
        steps_survived: u64,
        /// IEEE-754 bits of the exact queued final activation.
        final_activation_bits: u64,
        /// IEEE-754 bits of the exact queued Psi value.
        psi_bits: u64,
        /// IEEE-754 bits of the exact queued coherence value.
        coherence_bits: u64,
        /// Legacy source classification queued with the candidate.
        source: MemorySource,
        /// Legacy verification flag queued with the candidate.
        is_verified: bool,
    },
    /// Replace the complete active cognitive-state owner commitment.
    ///
    /// For the first hook these digests should come from [`active_state_digest_v1`],
    /// which commits to both the full `LiquidHolocell` and `current_thought`.
    ActiveStateReplace {
        /// Commitment to the complete pre-transition active owner state.
        before: Digest32,
        /// Commitment to the complete resulting active owner state.
        after: Digest32,
    },
    /// Append one exact goal record to the goal store.
    GoalActivate {
        /// Exact generated goal identifier.
        goal_id: String,
        /// Exact legacy goal description.
        description: String,
        /// Commitment to the exact goal embedding.
        embedding: Digest32,
        /// IEEE-754 bits of the exact stored priority.
        priority_bits: u32,
        /// IEEE-754 bits of the exact stored progress.
        progress_bits: u32,
        /// Exact stored active flag.
        is_active: bool,
    },
    /// Apply one emotional-valence transition.
    AffectSet {
        /// IEEE-754 bits of the valence before the transition.
        before_bits: u32,
        /// IEEE-754 bits of the parsed input delta used by legacy code.
        delta_bits: u32,
        /// IEEE-754 bits of the exact post-clamp result.
        after_bits: u32,
    },
}

impl CognitiveEffectV1 {
    /// Build a graduation effect from the exact float values queued by legacy code.
    #[allow(clippy::too_many_arguments)]
    pub fn graduation_enqueue(
        content: Digest32,
        label: impl Into<String>,
        steps_survived: u64,
        final_activation: f64,
        psi: f64,
        coherence: f64,
        source: MemorySource,
        is_verified: bool,
    ) -> Self {
        Self::GraduationEnqueue {
            content,
            label: label.into(),
            steps_survived,
            final_activation_bits: final_activation.to_bits(),
            psi_bits: psi.to_bits(),
            coherence_bits: coherence.to_bits(),
            source,
            is_verified,
        }
    }

    /// Build an active-state replacement from the exact complete owner states.
    pub fn active_state_replace(
        before_holocell: &LiquidHolocell,
        before_current_thought: &ContinuousHV,
        after_holocell: &LiquidHolocell,
        after_current_thought: &ContinuousHV,
    ) -> Self {
        Self::ActiveStateReplace {
            before: active_state_digest_v1(before_holocell, before_current_thought),
            after: active_state_digest_v1(after_holocell, after_current_thought),
        }
    }

    /// Build a goal-activation effect from the exact values written to the goal store.
    pub fn goal_activate(
        goal_id: impl Into<String>,
        description: impl Into<String>,
        embedding: Digest32,
        priority: f32,
        progress: f32,
        is_active: bool,
    ) -> Self {
        Self::GoalActivate {
            goal_id: goal_id.into(),
            description: description.into(),
            embedding,
            priority_bits: priority.to_bits(),
            progress_bits: progress.to_bits(),
            is_active,
        }
    }

    /// Build an affect transition from the exact legacy before/delta/after values.
    pub fn affect_set(before: f32, delta: f32, after: f32) -> Self {
        Self::AffectSet {
            before_bits: before.to_bits(),
            delta_bits: delta.to_bits(),
            after_bits: after.to_bits(),
        }
    }
}

/// Commit to one `ContinuousHV` using exact component bit patterns in index order.
///
/// The commitment is intentionally bit-exact: `+0.0` and `-0.0`, and distinct
/// NaN payloads, produce different commitments because they are different stored
/// representations. A future semantic-normalization rule must use a new schema/domain.
pub fn continuous_hv_digest_v1(hv: &ContinuousHV) -> Digest32 {
    let mut out = CanonicalWriter::with_domain(HDC_DOMAIN_V1);
    out.u64(hv.values.len() as u64);
    for value in &hv.values {
        out.u32(value.to_bits());
    }
    sha256(&out.finish())
}

/// Commit to the complete `LiquidHolocell` representation used by `ContinuousMind`.
///
/// The dimensionality enum is encoded by an explicit frozen discriminant rather
/// than serde/Rust layout. `Custom(n)` remains distinct from a predefined variant
/// even when `n` numerically equals one of the predefined dimensions.
pub fn liquid_holocell_digest_v1(holocell: &LiquidHolocell) -> Digest32 {
    let mut out = CanonicalWriter::with_domain(HOLOCELL_DOMAIN_V1);
    out.digest(continuous_hv_digest_v1(&holocell.state));
    out.u32(holocell.tau.to_bits());
    encode_dimensionality(&mut out, holocell.dimensionality);
    out.u32(holocell.pressure.to_bits());
    sha256(&out.finish())
}

/// Commit to the complete first-hook active-state owner representation.
///
/// `process_inputs()` mutates the `LiquidHolocell` and then copies its state into
/// `current_thought`. Both writes are committed so an accidental divergence
/// between those fields cannot share an effect identity.
pub fn active_state_digest_v1(
    holocell: &LiquidHolocell,
    current_thought: &ContinuousHV,
) -> Digest32 {
    let mut out = CanonicalWriter::with_domain(ACTIVE_STATE_DOMAIN_V1);
    out.digest(liquid_holocell_digest_v1(holocell));
    out.digest(continuous_hv_digest_v1(current_thought));
    sha256(&out.finish())
}

/// Commit to a legacy metadata map in deterministic UTF-8 key order.
pub fn metadata_digest_v1(metadata: &HashMap<String, String>) -> Digest32 {
    let mut entries: Vec<(&String, &String)> = metadata.iter().collect();
    entries.sort_by(|(ka, va), (kb, vb)| {
        ka.as_bytes()
            .cmp(kb.as_bytes())
            .then_with(|| va.as_bytes().cmp(vb.as_bytes()))
    });

    let mut out = CanonicalWriter::with_domain(METADATA_DOMAIN_V1);
    out.u64(entries.len() as u64);
    for (key, value) in entries {
        out.string(key);
        out.string(value);
    }
    sha256(&out.finish())
}

/// Return the exact canonical v1 byte representation of one cognitive effect.
pub fn canonical_effect_bytes_v1(effect: &CognitiveEffectV1) -> Vec<u8> {
    let mut out = CanonicalWriter::with_domain(EFFECT_DOMAIN_V1);
    match effect {
        CognitiveEffectV1::WorkingMemoryAdmit {
            item,
            arrival_tick,
            source,
            verified,
            metadata,
            insertion_index,
        } => {
            out.u8(EFFECT_WM_ADMIT);
            out.digest(*item);
            out.u64(*arrival_tick);
            out.u8(memory_source_code(*source));
            out.bool(*verified);
            out.digest(*metadata);
            out.u64(*insertion_index);
        }
        CognitiveEffectV1::WorkingMemoryReplace {
            admitted_item,
            admitted_arrival_tick,
            admitted_source,
            admitted_verified,
            admitted_metadata,
            admitted_index,
            evicted_item,
            evicted_arrival_tick,
            evicted_steps_survived,
            evicted_source,
            evicted_verified,
            evicted_metadata,
            evicted_index,
        } => {
            out.u8(EFFECT_WM_REPLACE);
            out.digest(*admitted_item);
            out.u64(*admitted_arrival_tick);
            out.u8(memory_source_code(*admitted_source));
            out.bool(*admitted_verified);
            out.digest(*admitted_metadata);
            out.u64(*admitted_index);
            out.digest(*evicted_item);
            out.u64(*evicted_arrival_tick);
            out.u64(*evicted_steps_survived);
            out.u8(memory_source_code(*evicted_source));
            out.bool(*evicted_verified);
            out.digest(*evicted_metadata);
            out.u64(*evicted_index);
        }
        CognitiveEffectV1::GraduationEnqueue {
            content,
            label,
            steps_survived,
            final_activation_bits,
            psi_bits,
            coherence_bits,
            source,
            is_verified,
        } => {
            out.u8(EFFECT_GRADUATION_ENQUEUE);
            out.digest(*content);
            out.string(label);
            out.u64(*steps_survived);
            out.u64(*final_activation_bits);
            out.u64(*psi_bits);
            out.u64(*coherence_bits);
            out.u8(memory_source_code(*source));
            out.bool(*is_verified);
        }
        CognitiveEffectV1::ActiveStateReplace { before, after } => {
            out.u8(EFFECT_ACTIVE_STATE_REPLACE);
            out.digest(*before);
            out.digest(*after);
        }
        CognitiveEffectV1::GoalActivate {
            goal_id,
            description,
            embedding,
            priority_bits,
            progress_bits,
            is_active,
        } => {
            out.u8(EFFECT_GOAL_ACTIVATE);
            out.string(goal_id);
            out.string(description);
            out.digest(*embedding);
            out.u32(*priority_bits);
            out.u32(*progress_bits);
            out.bool(*is_active);
        }
        CognitiveEffectV1::AffectSet {
            before_bits,
            delta_bits,
            after_bits,
        } => {
            out.u8(EFFECT_AFFECT_SET);
            out.u32(*before_bits);
            out.u32(*delta_bits);
            out.u32(*after_bits);
        }
    }
    out.finish()
}

/// Compute the canonical SHA-256 commitment for one v1 cognitive effect.
pub fn effect_digest_v1(effect: &CognitiveEffectV1) -> Digest32 {
    sha256(&canonical_effect_bytes_v1(effect))
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

fn encode_dimensionality(out: &mut CanonicalWriter, dimensionality: HdcDimensionality) {
    match dimensionality {
        HdcDimensionality::Rest => out.u8(0),
        HdcDimensionality::Standard => out.u8(1),
        HdcDimensionality::Extended => out.u8(2),
        HdcDimensionality::Ultra => out.u8(3),
        HdcDimensionality::Custom(dim) => {
            out.u8(4);
            out.u64(dim as u64);
        }
    }
}

fn sha256(bytes: &[u8]) -> Digest32 {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    Digest32(digest)
}

#[derive(Debug)]
struct CanonicalWriter {
    bytes: Vec<u8>,
}

impl CanonicalWriter {
    fn with_domain(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 1 + 128);
        bytes.extend_from_slice(domain);
        bytes.push(0);
        Self { bytes }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.bytes.extend_from_slice(&value.0);
    }

    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn admit() -> CognitiveEffectV1 {
        CognitiveEffectV1::WorkingMemoryAdmit {
            item: d(1),
            arrival_tick: 7,
            source: MemorySource::UserInteraction,
            verified: false,
            metadata: d(2),
            insertion_index: 3,
        }
    }

    fn replacement() -> CognitiveEffectV1 {
        CognitiveEffectV1::WorkingMemoryReplace {
            admitted_item: d(1),
            admitted_arrival_tick: 11,
            admitted_source: MemorySource::UserInteraction,
            admitted_verified: false,
            admitted_metadata: d(2),
            admitted_index: 3,
            evicted_item: d(9),
            evicted_arrival_tick: 4,
            evicted_steps_survived: 7,
            evicted_source: MemorySource::Internal,
            evicted_verified: false,
            evicted_metadata: d(8),
            evicted_index: 0,
        }
    }

    #[test]
    fn metadata_commitment_is_insertion_order_independent() {
        let mut a = HashMap::new();
        a.insert("topic".to_string(), "alpha".to_string());
        a.insert("source".to_string(), "user".to_string());

        let mut b = HashMap::new();
        b.insert("source".to_string(), "user".to_string());
        b.insert("topic".to_string(), "alpha".to_string());

        assert_eq!(metadata_digest_v1(&a), metadata_digest_v1(&b));
    }

    #[test]
    fn metadata_commitment_binds_keys_and_values() {
        let mut base = HashMap::new();
        base.insert("topic".to_string(), "alpha".to_string());

        let mut value_changed = base.clone();
        value_changed.insert("topic".to_string(), "beta".to_string());

        let mut key_changed = HashMap::new();
        key_changed.insert("subject".to_string(), "alpha".to_string());

        assert_ne!(metadata_digest_v1(&base), metadata_digest_v1(&value_changed));
        assert_ne!(metadata_digest_v1(&base), metadata_digest_v1(&key_changed));
    }

    #[test]
    fn hdc_commitment_is_bit_exact() {
        let positive_zero = ContinuousHV::from_values(vec![0.0, 1.0]);
        let negative_zero = ContinuousHV::from_values(vec![-0.0, 1.0]);

        assert_ne!(
            continuous_hv_digest_v1(&positive_zero),
            continuous_hv_digest_v1(&negative_zero)
        );
    }

    #[test]
    fn holocell_commitment_binds_all_fields_and_variant_identity() {
        let base = LiquidHolocell {
            state: ContinuousHV::from_values(vec![0.25, -0.5]),
            tau: 1.0,
            dimensionality: HdcDimensionality::Standard,
            pressure: 0.2,
        };
        let mut tau_changed = base.clone();
        tau_changed.tau = 2.0;
        let mut pressure_changed = base.clone();
        pressure_changed.pressure = 0.3;
        let mut variant_changed = base.clone();
        variant_changed.dimensionality = HdcDimensionality::Custom(16_384);

        assert_ne!(liquid_holocell_digest_v1(&base), liquid_holocell_digest_v1(&tau_changed));
        assert_ne!(
            liquid_holocell_digest_v1(&base),
            liquid_holocell_digest_v1(&pressure_changed)
        );
        assert_ne!(
            liquid_holocell_digest_v1(&base),
            liquid_holocell_digest_v1(&variant_changed)
        );
    }

    #[test]
    fn active_state_commitment_binds_current_thought_separately() {
        let cell = LiquidHolocell {
            state: ContinuousHV::from_values(vec![0.25, -0.5]),
            tau: 1.0,
            dimensionality: HdcDimensionality::Custom(2),
            pressure: 0.0,
        };
        let thought_a = cell.state.clone();
        let thought_b = ContinuousHV::from_values(vec![0.25, -0.4]);

        assert_ne!(
            active_state_digest_v1(&cell, &thought_a),
            active_state_digest_v1(&cell, &thought_b)
        );
    }

    #[test]
    fn admit_and_replace_are_domain_distinct() {
        assert_ne!(effect_digest_v1(&admit()), effect_digest_v1(&replacement()));
    }

    #[test]
    fn working_memory_commitment_binds_arrival_tick_and_legacy_flags() {
        let base = admit();
        let mut tick_changed = base.clone();
        if let CognitiveEffectV1::WorkingMemoryAdmit { arrival_tick, .. } = &mut tick_changed {
            *arrival_tick += 1;
        }
        let mut verified_changed = base.clone();
        if let CognitiveEffectV1::WorkingMemoryAdmit { verified, .. } = &mut verified_changed {
            *verified = true;
        }

        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&tick_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&verified_changed));
    }

    #[test]
    fn replacement_binds_exact_eviction_target_and_timing() {
        let base = replacement();
        let mut target_changed = base.clone();
        if let CognitiveEffectV1::WorkingMemoryReplace { evicted_item, .. } = &mut target_changed {
            *evicted_item = d(10);
        }
        let mut timing_changed = base.clone();
        if let CognitiveEffectV1::WorkingMemoryReplace {
            evicted_steps_survived,
            ..
        } = &mut timing_changed
        {
            *evicted_steps_survived += 1;
        }

        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&target_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&timing_changed));
    }

    #[test]
    fn graduation_commitment_binds_label_float_source_and_verification() {
        let base = CognitiveEffectV1::graduation_enqueue(
            d(3),
            "topic",
            5,
            0.5,
            0.4,
            0.4,
            MemorySource::Internal,
            false,
        );
        let label_changed = CognitiveEffectV1::graduation_enqueue(
            d(3),
            "other",
            5,
            0.5,
            0.4,
            0.4,
            MemorySource::Internal,
            false,
        );
        let psi_changed = CognitiveEffectV1::graduation_enqueue(
            d(3),
            "topic",
            5,
            0.5,
            0.41,
            0.4,
            MemorySource::Internal,
            false,
        );
        let source_changed = CognitiveEffectV1::graduation_enqueue(
            d(3),
            "topic",
            5,
            0.5,
            0.4,
            0.4,
            MemorySource::WebResearch,
            false,
        );
        let verified_changed = CognitiveEffectV1::graduation_enqueue(
            d(3),
            "topic",
            5,
            0.5,
            0.4,
            0.4,
            MemorySource::Internal,
            true,
        );

        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&label_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&psi_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&source_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&verified_changed));
    }

    #[test]
    fn goal_commitment_binds_description_and_priority() {
        let base = CognitiveEffectV1::goal_activate(
            "goal_0",
            "protect evidence",
            d(4),
            0.8,
            0.0,
            true,
        );
        let description_changed = CognitiveEffectV1::goal_activate(
            "goal_0",
            "rewrite evidence",
            d(4),
            0.8,
            0.0,
            true,
        );
        let priority_changed = CognitiveEffectV1::goal_activate(
            "goal_0",
            "protect evidence",
            d(4),
            0.7,
            0.0,
            true,
        );

        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&description_changed));
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&priority_changed));
    }

    #[test]
    fn affect_commitment_binds_exact_result() {
        let base = CognitiveEffectV1::affect_set(0.2, 0.5, 0.35);
        let changed = CognitiveEffectV1::affect_set(0.2, 0.5, 0.36);
        assert_ne!(effect_digest_v1(&base), effect_digest_v1(&changed));
    }

    #[test]
    fn memory_source_has_stable_effect_discriminants() {
        let internal = CognitiveEffectV1::WorkingMemoryAdmit {
            item: d(1),
            arrival_tick: 7,
            source: MemorySource::Internal,
            verified: false,
            metadata: d(2),
            insertion_index: 0,
        };
        let social = CognitiveEffectV1::WorkingMemoryAdmit {
            item: d(1),
            arrival_tick: 7,
            source: MemorySource::Social,
            verified: false,
            metadata: d(2),
            insertion_index: 0,
        };
        assert_ne!(effect_digest_v1(&internal), effect_digest_v1(&social));
    }

    #[test]
    fn canonical_bytes_are_deterministic_for_same_effect() {
        let effect = replacement();
        assert_eq!(
            canonical_effect_bytes_v1(&effect),
            canonical_effect_bytes_v1(&effect)
        );
    }

    #[test]
    fn frozen_affect_test_vector_prevents_encoder_drift() {
        let effect = CognitiveEffectV1::AffectSet {
            before_bits: 0x3f80_0000,
            delta_bits: 0xbf00_0000,
            after_bits: 0x3f00_0000,
        };
        assert_eq!(
            effect_digest_v1(&effect),
            Digest32([
                100, 46, 80, 133, 81, 212, 42, 0, 239, 61, 87, 217, 52, 208, 207, 199,
                231, 82, 201, 68, 234, 219, 98, 61, 70, 217, 209, 144, 8, 246, 214, 218,
            ])
        );
    }
}
