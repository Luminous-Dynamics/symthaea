// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, resource-bounded representation of an IIT 4.0 macro unit.
//!
//! This module does not yet construct a macro TPM or judge the intrinsic-unit
//! postulates. It freezes the identity and truth-table semantics that those
//! later computations must consume.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};
use thiserror::Error;

const MACRO_MAPPING_MAGIC: &[u8] = b"SYMTHAEA-IIT-MACRO-MAPPING\0v0.1\0";
const MACRO_UNIT_SPEC_MAGIC: &[u8] = b"SYMTHAEA-IIT-MACRO-UNIT-SPEC\0v0.2\0";
const MACRO_MAPPING_SEARCH_MAGIC: &[u8] =
    b"SYMTHAEA-IIT-MACRO-MAPPING-SEARCH\0v0.1\0";

/// Errors for an invalid or over-budget macro-unit mapping.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MacroUnitError {
    #[error("a macro unit must have at least one constituent")]
    EmptyConstituents,
    #[error("macro-unit constituents must be strictly increasing; duplicate unit {unit}")]
    DuplicateConstituent { unit: usize },
    #[error("macro-unit constituents must be strictly increasing; {previous} precedes {current}")]
    UnsortedConstituents { previous: usize, current: usize },
    #[error("macro-unit update grain must be positive")]
    ZeroUpdateGrain,
    #[error("macro-unit sequence-state domain overflows usize")]
    SequenceDomainOverflow,
    #[error("macro-unit mapping requires {required} entries, exceeding budget {limit}")]
    MappingOverBudget { required: usize, limit: usize },
    #[error("macro-unit mapping length mismatch: expected {expected}, got {actual}")]
    MappingLengthMismatch { expected: usize, actual: usize },
    #[error("on-count {count} is outside 0..={max}")]
    InvalidOnCount { count: usize, max: usize },
    #[error("blackboxing requires at least one output constituent")]
    EmptyBlackboxOutputs,
    #[error("blackbox output constituent {slot} is repeated")]
    DuplicateBlackboxOutput { slot: usize },
    #[error("blackbox output constituent {slot} is outside 0..{constituents}")]
    BlackboxOutputOutOfRange { slot: usize, constituents: usize },
    #[error("background apportionment contains duplicate micro unit {unit}")]
    DuplicateBackgroundApportionment { unit: usize },
    #[error("background apportionment overlaps constituent micro unit {unit}")]
    BackgroundOverlapsConstituents { unit: usize },
    #[error("default mapping-family candidate count overflows usize")]
    CandidateCountOverflow,
    #[error("mapping-family search requires at most {required} candidates, exceeding budget {limit}")]
    CandidateBudgetExceeded { required: usize, limit: usize },
    #[error("mapping-family search requires at most {required} truth-table entries, exceeding budget {limit}")]
    TotalMappingEntryBudgetExceeded { required: usize, limit: usize },
    #[error("constant mappings do not define a non-trivial macro unit")]
    ConstantMapping,
    #[error("sequence-state index {index} is outside the mapping domain of {states} states")]
    SequenceStateOutOfRange { index: usize, states: usize },
    #[error("micro-history length mismatch: expected {expected} oldest-first states, got {actual}")]
    HistoryLengthMismatch { expected: usize, actual: usize },
    #[error("micro-history state width mismatch: expected {expected}, got {actual}")]
    HistoryStateWidthMismatch { expected: usize, actual: usize },
    #[error("micro-history state is too narrow for constituent {unit}: width {actual}")]
    HistoryStateTooNarrow { unit: usize, actual: usize },
}

/// Explicit preflight limits for default coarse-grain/blackbox mapping enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroMappingSearchLimits {
    /// Maximum raw candidate mappings before duplicate/complement elimination.
    pub max_candidate_upper_bound: usize,
    /// Maximum aggregate entries across the preflight upper bound.
    pub max_total_mapping_entries: usize,
    /// Maximum entries in any single mapping truth table.
    pub max_mapping_entries: usize,
}

/// Complete default-family mapping enumeration with deterministic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroMappingSearchReport {
    /// Candidate mappings in PyPhi-compatible first-seen order.
    pub candidates: Vec<MacroUnitSpec>,
    /// Number of raw ON-count coarse-graining tables considered.
    pub generated_coarse_grain: usize,
    /// Number of raw blackbox output-subset tables considered.
    pub generated_blackbox: usize,
    /// Number of duplicate canonical mappings removed.
    pub duplicates_removed: usize,
    /// Raw candidate upper bound checked before allocation.
    pub candidate_upper_bound: usize,
    /// Aggregate truth-table entries in the deduplicated output.
    pub total_mapping_entries: usize,
    /// Digest of the complete candidate set, bounds and generation policy.
    pub candidate_set_sha256: String,
    /// Always true for a successfully returned report.
    pub complete: bool,
}

/// A binary macro unit with an explicit constituent order, temporal grain,
/// truth table, background apportionment, and complement orientation.
///
/// The truth table's input index is a little-endian bit index. Bit
/// (time * constituents.len() + constituent_slot) selects the named
/// constituent at that time step; time is oldest-first and the first
/// constituent is the least-significant bit of the oldest micro state.
///
/// A mapping and its Boolean complement describe the same unoriented macro
/// unit with opposite macro-state labels. mapping_sha256 therefore binds the
/// canonical representative (the lexicographically smaller of the mapping and
/// its complement), while spec_sha256 additionally binds background
/// apportionment and the orientation. This keeps unit-shape equivalence
/// separate from the state-label transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroUnitSpec {
    constituents: Vec<usize>,
    update_grain: usize,
    sequence_state_bits: usize,
    canonical_mapping: Vec<bool>,
    background_apportionment: Vec<usize>,
    complement_flipped: bool,
    mapping_sha256: String,
    spec_sha256: String,
}

impl MacroUnitSpec {
    /// Enumerate PyPhi-compatible default mapping families under explicit budgets.
    ///
    /// At update grain 1, this emits every non-degenerate ON-count coarse
    /// graining, followed by every non-empty blackbox output subset. At larger
    /// grains only blackbox families are emitted. Tables are canonicalized so
    /// the all-OFF sequence maps to macro-state 0, then deduplicated while
    /// preserving first-seen family order.
    ///
    /// The raw candidate upper bound and aggregate table-entry upper bound are
    /// checked before candidate tables are allocated. This can conservatively
    /// reject a search whose eventual de-duplicated set would fit; it never
    /// allocates beyond the caller's declared limits.
    pub fn enumerate_default_mapping_families(
        constituents: Vec<usize>,
        update_grain: usize,
        background_apportionment: Vec<usize>,
        limits: MacroMappingSearchLimits,
    ) -> Result<MacroMappingSearchReport, MacroUnitError> {
        if constituents.is_empty() {
            return Err(MacroUnitError::EmptyConstituents);
        }
        for pair in constituents.windows(2) {
            if pair[0] == pair[1] {
                return Err(MacroUnitError::DuplicateConstituent { unit: pair[0] });
            }
            if pair[0] > pair[1] {
                return Err(MacroUnitError::UnsortedConstituents {
                    previous: pair[0],
                    current: pair[1],
                });
            }
        }
        if update_grain == 0 {
            return Err(MacroUnitError::ZeroUpdateGrain);
        }
        let (_, mapping_entries) =
            mapping_domain(constituents.len(), update_grain, limits.max_mapping_entries)?;
        let mut background_apportionment = background_apportionment;
        background_apportionment.sort_unstable();
        for pair in background_apportionment.windows(2) {
            if pair[0] == pair[1] {
                return Err(MacroUnitError::DuplicateBackgroundApportionment { unit: pair[0] });
            }
        }
        for unit in &background_apportionment {
            if constituents.binary_search(unit).is_ok() {
                return Err(MacroUnitError::BackgroundOverlapsConstituents { unit: *unit });
            }
        }

        let n = constituents.len();
        let blackbox_mask_count = 1usize
            .checked_shl(n as u32)
            .ok_or(MacroUnitError::CandidateCountOverflow)?;
        let blackbox_upper = blackbox_mask_count - 1;
        let coarse_upper = if update_grain == 1 {
            1usize
                .checked_shl(
                    n.checked_add(1)
                        .ok_or(MacroUnitError::CandidateCountOverflow)? as u32,
                )
                .ok_or(MacroUnitError::CandidateCountOverflow)?
                .checked_sub(2)
                .ok_or(MacroUnitError::CandidateCountOverflow)?
        } else {
            0
        };
        let candidate_upper_bound = coarse_upper
            .checked_add(blackbox_upper)
            .ok_or(MacroUnitError::CandidateCountOverflow)?;
        if candidate_upper_bound > limits.max_candidate_upper_bound {
            return Err(MacroUnitError::CandidateBudgetExceeded {
                required: candidate_upper_bound,
                limit: limits.max_candidate_upper_bound,
            });
        }
        let total_upper_entries = candidate_upper_bound
            .checked_mul(mapping_entries)
            .ok_or(MacroUnitError::CandidateCountOverflow)?;
        if total_upper_entries > limits.max_total_mapping_entries {
            return Err(MacroUnitError::TotalMappingEntryBudgetExceeded {
                required: total_upper_entries,
                limit: limits.max_total_mapping_entries,
            });
        }

        let state_mask = 1usize
            .checked_shl(n as u32)
            .ok_or(MacroUnitError::SequenceDomainOverflow)?
            - 1;
        let final_update_offset = (update_grain - 1)
            .checked_mul(n)
            .ok_or(MacroUnitError::SequenceDomainOverflow)?;
        let mut seen = BTreeSet::<String>::new();
        let mut candidates = Vec::with_capacity(candidate_upper_bound);
        let mut generated_coarse_grain = 0usize;
        let mut generated_blackbox = 0usize;

        let mut add_mapping = |mapping: Vec<bool>| -> Result<(), MacroUnitError> {
            let canonical = if mapping[0] {
                mapping.into_iter().map(|value| !value).collect::<Vec<_>>()
            } else {
                mapping
            };
            let mapping_sha256 =
                Self::mapping_digest(&constituents, update_grain, &canonical);
            if seen.insert(mapping_sha256) {
                candidates.push(Self::new_with_background(
                    constituents.clone(),
                    update_grain,
                    canonical,
                    background_apportionment.clone(),
                    limits.max_mapping_entries,
                )?);
            }
            Ok(())
        };

        if update_grain == 1 {
            let count_options = n
                .checked_add(1)
                .ok_or(MacroUnitError::CandidateCountOverflow)?;
            let subset_limit = 1usize
                .checked_shl(count_options as u32)
                .ok_or(MacroUnitError::CandidateCountOverflow)?;
            for subset_size in 1..count_options {
                for on_count_mask in 1..subset_limit - 1 {
                    if on_count_mask.count_ones() as usize != subset_size {
                        continue;
                    }
                    let mut mapping = Vec::with_capacity(mapping_entries);
                    for state in 0..mapping_entries {
                        let on_count = (state & state_mask).count_ones() as usize;
                        mapping.push(((on_count_mask >> on_count) & 1) != 0);
                    }
                    generated_coarse_grain = generated_coarse_grain
                        .checked_add(1)
                        .ok_or(MacroUnitError::CandidateCountOverflow)?;
                    add_mapping(mapping)?;
                }
            }
        }

        for output_size in 1..=n {
            for output_mask in 1..blackbox_mask_count {
                if output_mask.count_ones() as usize != output_size {
                    continue;
                }
                let mut mapping = Vec::with_capacity(mapping_entries);
                for sequence_state in 0..mapping_entries {
                    let final_state = (sequence_state >> final_update_offset) & state_mask;
                    mapping.push((output_mask & final_state) == output_mask);
                }
                generated_blackbox = generated_blackbox
                    .checked_add(1)
                    .ok_or(MacroUnitError::CandidateCountOverflow)?;
                add_mapping(mapping)?;
            }
        }

        let duplicates_removed = candidate_upper_bound
            .checked_sub(candidates.len())
            .ok_or(MacroUnitError::CandidateCountOverflow)?;
        let total_mapping_entries = candidates
            .len()
            .checked_mul(mapping_entries)
            .ok_or(MacroUnitError::CandidateCountOverflow)?;
        let candidate_set_sha256 = Self::candidate_set_digest(
            &constituents,
            update_grain,
            &background_apportionment,
            limits,
            generated_coarse_grain,
            generated_blackbox,
            duplicates_removed,
            &candidates,
        );

        let report = MacroMappingSearchReport {
            candidates,
            generated_coarse_grain,
            generated_blackbox,
            duplicates_removed,
            candidate_upper_bound,
            total_mapping_entries,
            candidate_set_sha256,
            complete: true,
        };
        debug_assert_eq!(
            report.candidate_upper_bound,
            report.generated_coarse_grain + report.generated_blackbox
        );
        Ok(report)
    }

    fn candidate_set_digest(
        constituents: &[usize],
        update_grain: usize,
        background_apportionment: &[usize],
        limits: MacroMappingSearchLimits,
        coarse_count: usize,
        blackbox_count: usize,
        duplicates_removed: usize,
        candidates: &[MacroUnitSpec],
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MACRO_MAPPING_SEARCH_MAGIC);
        bytes.extend_from_slice(&(constituents.len() as u64).to_le_bytes());
        for unit in constituents {
            bytes.extend_from_slice(&(*unit as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(update_grain as u64).to_le_bytes());
        bytes.extend_from_slice(&(background_apportionment.len() as u64).to_le_bytes());
        for unit in background_apportionment {
            bytes.extend_from_slice(&(*unit as u64).to_le_bytes());
        }
        for limit in [
            limits.max_candidate_upper_bound,
            limits.max_total_mapping_entries,
            limits.max_mapping_entries,
        ] {
            bytes.extend_from_slice(&(limit as u64).to_le_bytes());
        }
        for count in [coarse_count, blackbox_count, duplicates_removed, candidates.len()] {
            bytes.extend_from_slice(&(count as u64).to_le_bytes());
        }
        for candidate in candidates {
            bytes.extend_from_slice(candidate.spec_sha256.as_bytes());
        }
        Self::hex_digest(&bytes)
    }

    /// Build the PyPhi-compatible coarse-graining family at update grain 1.
    ///
    /// A macro state is ON when the number of ON constituents belongs to
    /// on_counts. The sequence-state table uses the same little-endian order
    /// as the general MacroUnitSpec constructor.
    pub fn coarse_grain(
        constituents: Vec<usize>,
        on_counts: &[usize],
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        Self::coarse_grain_with_background(
            constituents,
            on_counts,
            Vec::new(),
            max_mapping_entries,
        )
    }

    /// Build a coarse-graining unit with explicit background apportionment.
    pub fn coarse_grain_with_background(
        constituents: Vec<usize>,
        on_counts: &[usize],
        background_apportionment: Vec<usize>,
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        let maximum_count = constituents.len();
        for count in on_counts {
            if *count > maximum_count {
                return Err(MacroUnitError::InvalidOnCount {
                    count: *count,
                    max: maximum_count,
                });
            }
        }
        let (_, expected_entries) =
            mapping_domain(constituents.len(), 1, max_mapping_entries)?;
        let on_count_set = on_counts.iter().copied().collect::<std::collections::BTreeSet<_>>();
        let mapping = (0..expected_entries)
            .map(|state| on_count_set.contains(&(state.count_ones() as usize)))
            .collect();
        Self::new_with_background(
            constituents,
            1,
            mapping,
            background_apportionment,
            max_mapping_entries,
        )
    }

    /// Build the PyPhi-compatible blackboxing family.
    ///
    /// The macro state is ON exactly when every designated output constituent
    /// is ON at the final micro update of the temporal window. All earlier
    /// micro states and non-designated outputs are ignored by this mapping.
    pub fn blackbox(
        constituents: Vec<usize>,
        update_grain: usize,
        output_constituents: &[usize],
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        Self::blackbox_with_background(
            constituents,
            update_grain,
            output_constituents,
            Vec::new(),
            max_mapping_entries,
        )
    }

    /// Build a blackboxing unit with explicit background apportionment.
    pub fn blackbox_with_background(
        constituents: Vec<usize>,
        update_grain: usize,
        output_constituents: &[usize],
        background_apportionment: Vec<usize>,
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        if output_constituents.is_empty() {
            return Err(MacroUnitError::EmptyBlackboxOutputs);
        }
        let mut seen = std::collections::BTreeSet::new();
        for slot in output_constituents {
            if !seen.insert(*slot) {
                return Err(MacroUnitError::DuplicateBlackboxOutput { slot: *slot });
            }
            if *slot >= constituents.len() {
                return Err(MacroUnitError::BlackboxOutputOutOfRange {
                    slot: *slot,
                    constituents: constituents.len(),
                });
            }
        }
        let (_, expected_entries) =
            mapping_domain(constituents.len(), update_grain, max_mapping_entries)?;
        let last_update_offset = (update_grain - 1)
            .checked_mul(constituents.len())
            .ok_or(MacroUnitError::SequenceDomainOverflow)?;
        let mapping = (0..expected_entries)
            .map(|state| {
                output_constituents.iter().all(|slot| {
                    let bit = last_update_offset + *slot;
                    ((state >> bit) & 1) != 0
                })
            })
            .collect();
        Self::new_with_background(
            constituents,
            update_grain,
            mapping,
            background_apportionment,
            max_mapping_entries,
        )
    }

    /// Construct a macro unit without allocating more than max_mapping_entries.
    ///
    /// Constituents must already be in ascending order. Requiring an explicit
    /// order avoids silently permuting truth-table axes while canonicalizing a
    /// model identity.
    pub fn new(
        constituents: Vec<usize>,
        update_grain: usize,
        mapping: Vec<bool>,
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        Self::new_with_background(
            constituents,
            update_grain,
            mapping,
            Vec::new(),
            max_mapping_entries,
        )
    }

    /// Construct a macro unit with explicit background apportionment.
    ///
    /// Background indices are a set and are canonicalized in ascending order.
    /// They must be unique and disjoint from the unit's micro constituents.
    pub fn new_with_background(
        constituents: Vec<usize>,
        update_grain: usize,
        mapping: Vec<bool>,
        mut background_apportionment: Vec<usize>,
        max_mapping_entries: usize,
    ) -> Result<Self, MacroUnitError> {
        if constituents.is_empty() {
            return Err(MacroUnitError::EmptyConstituents);
        }
        for pair in constituents.windows(2) {
            if pair[0] == pair[1] {
                return Err(MacroUnitError::DuplicateConstituent { unit: pair[0] });
            }
            if pair[0] > pair[1] {
                return Err(MacroUnitError::UnsortedConstituents {
                    previous: pair[0],
                    current: pair[1],
                });
            }
        }
        if update_grain == 0 {
            return Err(MacroUnitError::ZeroUpdateGrain);
        }

        let (sequence_state_bits, expected_entries) =
            mapping_domain(constituents.len(), update_grain, max_mapping_entries)?;
        if mapping.len() != expected_entries {
            return Err(MacroUnitError::MappingLengthMismatch {
                expected: expected_entries,
                actual: mapping.len(),
            });
        }
        if mapping.iter().all(|value| *value == mapping[0]) {
            return Err(MacroUnitError::ConstantMapping);
        }

        background_apportionment.sort_unstable();
        for pair in background_apportionment.windows(2) {
            if pair[0] == pair[1] {
                return Err(MacroUnitError::DuplicateBackgroundApportionment { unit: pair[0] });
            }
        }
        for unit in &background_apportionment {
            if constituents.binary_search(unit).is_ok() {
                return Err(MacroUnitError::BackgroundOverlapsConstituents { unit: *unit });
            }
        }

        let complemented = mapping.iter().map(|value| !*value).collect::<Vec<_>>();
        let complement_flipped = complemented < mapping;
        let canonical_mapping = if complement_flipped {
            complemented
        } else {
            mapping
        };
        let mapping_sha256 = Self::mapping_digest(
            &constituents,
            update_grain,
            &canonical_mapping,
        );
        let spec_sha256 = Self::spec_digest(
            &mapping_sha256,
            complement_flipped,
            &background_apportionment,
        );

        Ok(Self {
            constituents,
            update_grain,
            sequence_state_bits,
            canonical_mapping,
            background_apportionment,
            complement_flipped,
            mapping_sha256,
            spec_sha256,
        })
    }

    fn mapping_digest(
        constituents: &[usize],
        update_grain: usize,
        mapping: &[bool],
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MACRO_MAPPING_MAGIC);
        bytes.extend_from_slice(&(constituents.len() as u64).to_le_bytes());
        for constituent in constituents {
            bytes.extend_from_slice(&(*constituent as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(update_grain as u64).to_le_bytes());
        bytes.extend_from_slice(&(mapping.len() as u64).to_le_bytes());
        bytes.extend(mapping.iter().map(|value| u8::from(*value)));
        Self::hex_digest(&bytes)
    }

    fn spec_digest(
        mapping_sha256: &str,
        complement_flipped: bool,
        background_apportionment: &[usize],
    ) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MACRO_UNIT_SPEC_MAGIC);
        bytes.extend_from_slice(mapping_sha256.as_bytes());
        bytes.push(u8::from(complement_flipped));
        bytes.extend_from_slice(&(background_apportionment.len() as u64).to_le_bytes());
        for unit in background_apportionment {
            bytes.extend_from_slice(&(*unit as u64).to_le_bytes());
        }
        Self::hex_digest(&bytes)
    }

    fn hex_digest(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Ordered micro-unit constituents.
    pub fn constituents(&self) -> &[usize] {
        &self.constituents
    }

    /// Number of consecutive micro updates read by this unit.
    pub const fn update_grain(&self) -> usize {
        self.update_grain
    }

    /// Number of input bits in one sequence-state.
    pub const fn sequence_state_bits(&self) -> usize {
        self.sequence_state_bits
    }

    /// Number of truth-table rows in the sequence-state domain.
    pub fn mapping_entries(&self) -> usize {
        self.canonical_mapping.len()
    }

    /// Canonical truth-table representative, shared by a mapping and its complement.
    pub fn canonical_mapping(&self) -> &[bool] {
        &self.canonical_mapping
    }

    /// Canonically ordered background micro-units apportioned to this macro unit.
    pub fn background_apportionment(&self) -> &[usize] {
        &self.background_apportionment
    }

    /// Whether the supplied truth table was the complement of the canonical representative.
    pub const fn complement_flipped(&self) -> bool {
        self.complement_flipped
    }

    /// Digest of the unoriented macro-unit mapping.
    pub fn mapping_sha256(&self) -> &str {
        &self.mapping_sha256
    }

    /// Digest of the mapping and its original macro-state orientation.
    pub fn spec_sha256(&self) -> &str {
        &self.spec_sha256
    }

    /// Apply the mapping to an oldest-first history of full micro-system states.
    ///
    /// The history must contain exactly update_grain states. Each state is
    /// indexed by global micro-unit ID; only this unit's constituent indices
    /// are read. The first listed constituent is the least-significant bit in
    /// each time slice, and the oldest time slice occupies the least-significant
    /// block of bits in the complete sequence-state index.
    pub fn map_micro_history(
        &self,
        history: &[Vec<bool>],
    ) -> Result<bool, MacroUnitError> {
        if history.len() != self.update_grain {
            return Err(MacroUnitError::HistoryLengthMismatch {
                expected: self.update_grain,
                actual: history.len(),
            });
        }
        let mut state_width = None;
        for state in history {
            if let Some(expected) = state_width {
                if state.len() != expected {
                    return Err(MacroUnitError::HistoryStateWidthMismatch {
                        expected,
                        actual: state.len(),
                    });
                }
            } else {
                state_width = Some(state.len());
            }
        }

        let maximum_constituent = self.constituents.last().copied().unwrap_or(0);
        let minimum_width = maximum_constituent
            .checked_add(1)
            .ok_or(MacroUnitError::SequenceDomainOverflow)?;
        let actual_width = state_width.unwrap_or(0);
        if actual_width < minimum_width {
            let unit = self
                .constituents
                .iter()
                .copied()
                .find(|unit| *unit >= actual_width)
                .unwrap_or(minimum_width - 1);
            return Err(MacroUnitError::HistoryStateTooNarrow {
                unit,
                actual: actual_width,
            });
        }

        let mut sequence_index = 0usize;
        for (time, state) in history.iter().enumerate() {
            for (slot, unit) in self.constituents.iter().copied().enumerate() {
                if state[unit] {
                    let bit = time
                        .checked_mul(self.constituents.len())
                        .and_then(|offset| offset.checked_add(slot))
                        .ok_or(MacroUnitError::SequenceDomainOverflow)?;
                    sequence_index |= 1usize
                        .checked_shl(bit as u32)
                        .ok_or(MacroUnitError::SequenceDomainOverflow)?;
                }
            }
        }
        self.map_sequence_state(sequence_index)
    }

    /// Evaluate the originally supplied truth table at one little-endian sequence-state index.
    pub fn map_sequence_state(&self, index: usize) -> Result<bool, MacroUnitError> {
        let Some(canonical_value) = self.canonical_mapping.get(index).copied() else {
            return Err(MacroUnitError::SequenceStateOutOfRange {
                index,
                states: self.canonical_mapping.len(),
            });
        };
        Ok(if self.complement_flipped {
            !canonical_value
        } else {
            canonical_value
        })
    }
}

fn mapping_domain(
    constituent_count: usize,
    update_grain: usize,
    max_mapping_entries: usize,
) -> Result<(usize, usize), MacroUnitError> {
    if constituent_count == 0 {
        return Err(MacroUnitError::EmptyConstituents);
    }
    if update_grain == 0 {
        return Err(MacroUnitError::ZeroUpdateGrain);
    }
    let sequence_state_bits = constituent_count
        .checked_mul(update_grain)
        .ok_or(MacroUnitError::SequenceDomainOverflow)?;
    if sequence_state_bits >= usize::BITS as usize {
        return Err(MacroUnitError::SequenceDomainOverflow);
    }
    let expected_entries = 1usize
        .checked_shl(sequence_state_bits as u32)
        .ok_or(MacroUnitError::SequenceDomainOverflow)?;
    if expected_entries > max_mapping_entries {
        return Err(MacroUnitError::MappingOverBudget {
            required: expected_entries,
            limit: max_mapping_entries,
        });
    }
    Ok((sequence_state_bits, expected_entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mapping_family_enumeration_matches_pinned_pyphi_counts() {
        let grain_one = MacroUnitSpec::enumerate_default_mapping_families(
            vec![0, 1],
            1,
            Vec::new(),
            MacroMappingSearchLimits {
                max_candidate_upper_bound: 9,
                max_total_mapping_entries: 36,
                max_mapping_entries: 4,
            },
        )
        .unwrap();
        assert_eq!(grain_one.generated_coarse_grain, 6);
        assert_eq!(grain_one.generated_blackbox, 3);
        assert_eq!(grain_one.candidate_upper_bound, 9);
        assert_eq!(grain_one.candidates.len(), 5);
        assert_eq!(grain_one.duplicates_removed, 4);
        assert_eq!(grain_one.total_mapping_entries, 20);
        assert_eq!(
            grain_one
                .candidates
                .iter()
                .map(|candidate| candidate.canonical_mapping().to_vec())
                .collect::<Vec<_>>(),
            vec![
                vec![false, true, true, true],
                vec![false, true, true, false],
                vec![false, false, false, true],
                vec![false, true, false, true],
                vec![false, false, true, true],
            ]
        );
        assert_eq!(
            grain_one.candidate_set_sha256,
            "833482091ba3267e40b747c67b5c6a89f133a99a871846cadd75b338a6dfe8bb"
        );
        assert!(grain_one.complete);

        let grain_two = MacroUnitSpec::enumerate_default_mapping_families(
            vec![0, 1],
            2,
            Vec::new(),
            MacroMappingSearchLimits {
                max_candidate_upper_bound: 3,
                max_total_mapping_entries: 48,
                max_mapping_entries: 16,
            },
        )
        .unwrap();
        assert_eq!(grain_two.generated_coarse_grain, 0);
        assert_eq!(grain_two.generated_blackbox, 3);
        assert_eq!(grain_two.candidate_upper_bound, 3);
        assert_eq!(grain_two.candidates.len(), 3);
        assert_eq!(grain_two.total_mapping_entries, 48);
        assert_eq!(
            grain_two.candidate_set_sha256,
            "ddc99e040f0f4cee463126406680b079a6420cd435f50d685cd634ca4338f879"
        );
        assert!(grain_two.complete);
    }

    #[test]
    fn default_mapping_family_digest_binds_background_and_is_order_canonical() {
        let limits = MacroMappingSearchLimits {
            max_candidate_upper_bound: 9,
            max_total_mapping_entries: 36,
            max_mapping_entries: 4,
        };
        let left = MacroUnitSpec::enumerate_default_mapping_families(
            vec![0, 1], 1, vec![3, 2], limits
        ).unwrap();
        let right = MacroUnitSpec::enumerate_default_mapping_families(
            vec![0, 1], 1, vec![2, 3], limits
        ).unwrap();
        assert_eq!(left.candidate_set_sha256, right.candidate_set_sha256);
        assert!(left
            .candidates
            .iter()
            .all(|candidate| candidate.background_apportionment() == [2, 3]));

        assert_eq!(
            MacroUnitSpec::enumerate_default_mapping_families(
                vec![0, 1],
                1,
                vec![2, 2],
                limits,
            ),
            Err(MacroUnitError::DuplicateBackgroundApportionment { unit: 2 })
        );
        assert_eq!(
            MacroUnitSpec::enumerate_default_mapping_families(
                vec![0, 1],
                1,
                vec![1],
                limits,
            ),
            Err(MacroUnitError::BackgroundOverlapsConstituents { unit: 1 })
        );
    }

    #[test]
    fn default_mapping_family_preflight_fails_before_candidate_allocation() {
        assert_eq!(
            MacroUnitSpec::enumerate_default_mapping_families(
                vec![0, 1],
                1,
                Vec::new(),
                MacroMappingSearchLimits {
                    max_candidate_upper_bound: 8,
                    max_total_mapping_entries: 36,
                    max_mapping_entries: 4,
                },
            ),
            Err(MacroUnitError::CandidateBudgetExceeded {
                required: 9,
                limit: 8
            })
        );
        assert_eq!(
            MacroUnitSpec::enumerate_default_mapping_families(
                vec![0, 1],
                1,
                Vec::new(),
                MacroMappingSearchLimits {
                    max_candidate_upper_bound: 9,
                    max_total_mapping_entries: 35,
                    max_mapping_entries: 4,
                },
            ),
            Err(MacroUnitError::TotalMappingEntryBudgetExceeded {
                required: 36,
                limit: 35
            })
        );
    }

    #[test]
    fn complemented_mappings_share_shape_digest_but_preserve_orientation() {
        let mapping = vec![false, false, true, true];
        let complement = mapping.iter().map(|value| !*value).collect::<Vec<_>>();
        let original = MacroUnitSpec::new(vec![0, 1], 1, mapping.clone(), 4).unwrap();
        let flipped = MacroUnitSpec::new(vec![0, 1], 1, complement, 4).unwrap();

        assert_eq!(original.mapping_sha256(), flipped.mapping_sha256());
        assert_ne!(original.spec_sha256(), flipped.spec_sha256());
        assert!(!original.complement_flipped());
        assert!(flipped.complement_flipped());
        assert_eq!(
            (0..4)
                .map(|index| original.map_sequence_state(index).unwrap())
                .collect::<Vec<_>>(),
            mapping
        );
        assert_eq!(
            (0..4)
                .map(|index| flipped.map_sequence_state(index).unwrap())
                .collect::<Vec<_>>(),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn temporal_mapping_has_explicit_oldest_first_sequence_domain() {
        let mapping = vec![
            false, true, false, true,
            false, true, false, true,
            false, true, false, true,
            false, true, false, true,
        ];
        let unit = MacroUnitSpec::new(vec![2, 5], 2, mapping, 16).unwrap();
        assert_eq!(unit.sequence_state_bits(), 4);
        assert_eq!(unit.mapping_entries(), 16);
        assert_eq!(unit.update_grain(), 2);
        assert_eq!(unit.constituents(), &[2, 5]);
        assert!(unit.map_sequence_state(15).unwrap());
        assert!(matches!(
            unit.map_sequence_state(16),
            Err(MacroUnitError::SequenceStateOutOfRange { index: 16, states: 16 })
        ));
    }

    #[test]
    fn rejects_invalid_and_over_budget_mapping_domains_before_construction() {
        assert_eq!(
            MacroUnitSpec::new(Vec::new(), 1, Vec::new(), 16),
            Err(MacroUnitError::EmptyConstituents)
        );
        assert_eq!(
            MacroUnitSpec::new(vec![1, 1], 1, vec![false; 4], 16),
            Err(MacroUnitError::DuplicateConstituent { unit: 1 })
        );
        assert_eq!(
            MacroUnitSpec::new(vec![2, 1], 1, vec![false; 4], 16),
            Err(MacroUnitError::UnsortedConstituents {
                previous: 2,
                current: 1
            })
        );
        assert_eq!(
            MacroUnitSpec::new(vec![0], 0, vec![false; 2], 16),
            Err(MacroUnitError::ZeroUpdateGrain)
        );
        assert_eq!(
            MacroUnitSpec::new(vec![0, 1, 2], 2, Vec::new(), 63),
            Err(MacroUnitError::MappingOverBudget {
                required: 64,
                limit: 63
            })
        );
        assert_eq!(
            MacroUnitSpec::new(vec![0], 1, vec![false, false], 2),
            Err(MacroUnitError::ConstantMapping)
        );
    }

    #[test]
    fn coarse_grain_matches_on_count_truth_table() {
        let unit = MacroUnitSpec::coarse_grain(vec![0, 1], &[2], 4).unwrap();
        assert_eq!(
            (0..4)
                .map(|index| unit.map_sequence_state(index).unwrap())
                .collect::<Vec<_>>(),
            vec![false, false, false, true]
        );

        let exactly_one = MacroUnitSpec::coarse_grain(vec![0, 1], &[1], 4).unwrap();
        assert_eq!(
            (0..4)
                .map(|index| exactly_one.map_sequence_state(index).unwrap())
                .collect::<Vec<_>>(),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn blackbox_reads_only_designated_outputs_at_final_update() {
        let unit = MacroUnitSpec::blackbox(vec![0, 1], 2, &[0], 16).unwrap();
        for state in 0..16 {
            let expected = ((state >> 2) & 1) != 0;
            assert_eq!(unit.map_sequence_state(state).unwrap(), expected);
        }

        let both_outputs = MacroUnitSpec::blackbox(vec![0, 1], 2, &[0, 1], 16).unwrap();
        for state in 0..16 {
            let expected = ((state >> 2) & 0b11) == 0b11;
            assert_eq!(both_outputs.map_sequence_state(state).unwrap(), expected);
        }
    }

    #[test]
    fn maps_micro_history_oldest_first_using_global_constituent_indices() {
        // The first constituent has global index 2 and the second index 5.
        // For ON-count=2 the macro state is ON only when both are ON in the
        // single oldest/current micro state.
        let unit = MacroUnitSpec::coarse_grain(vec![2, 5], &[2], 4).unwrap();
        let history = vec![vec![false, false, true, false, false, true]];
        assert!(unit.map_micro_history(&history).unwrap());

        let history = vec![vec![false, false, true, false, false, false]];
        assert!(!unit.map_micro_history(&history).unwrap());

        // Two-step blackbox reads the designated output only at the final
        // (newest) micro update. The oldest state is deliberately opposite.
        let blackbox = MacroUnitSpec::blackbox(vec![2, 5], 2, &[0], 16).unwrap();
        let history = vec![
            vec![false, false, true, false, false, false],
            vec![false, false, false, false, false, false],
        ];
        assert!(!blackbox.map_micro_history(&history).unwrap());
        let history = vec![
            vec![false, false, false, false, false, false],
            vec![false, false, true, false, false, false],
        ];
        assert!(blackbox.map_micro_history(&history).unwrap());
    }

    #[test]
    fn micro_history_rejects_incorrect_length_and_width_before_mapping() {
        let unit = MacroUnitSpec::blackbox(vec![2, 5], 2, &[0], 16).unwrap();
        assert_eq!(
            unit.map_micro_history(&[vec![false; 6]]),
            Err(MacroUnitError::HistoryLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            unit.map_micro_history(&[
                vec![false; 6],
                vec![false; 5]
            ]),
            Err(MacroUnitError::HistoryStateWidthMismatch {
                expected: 6,
                actual: 5
            })
        );
        assert_eq!(
            unit.map_micro_history(&[
                vec![false; 4],
                vec![false; 4]
            ]),
            Err(MacroUnitError::HistoryStateTooNarrow {
                unit: 5,
                actual: 4
            })
        );
    }

    #[test]
    fn micro_history_width_arithmetic_rejects_maximum_unit_index() {
        let unit = MacroUnitSpec::new(vec![usize::MAX], 1, vec![false, true], 2).unwrap();
        assert_eq!(
            unit.map_micro_history(&[Vec::new()]),
            Err(MacroUnitError::SequenceDomainOverflow)
        );
    }

    #[test]
    fn mapping_families_reject_invalid_local_controls() {
        assert!(matches!(
            MacroUnitSpec::coarse_grain(vec![0, 1], &[3], 4),
            Err(MacroUnitError::InvalidOnCount { count: 3, max: 2 })
        ));
        assert_eq!(
            MacroUnitSpec::blackbox(vec![0, 1], 1, &[], 4),
            Err(MacroUnitError::EmptyBlackboxOutputs)
        );
        assert_eq!(
            MacroUnitSpec::blackbox(vec![0, 1], 1, &[0, 0], 4),
            Err(MacroUnitError::DuplicateBlackboxOutput { slot: 0 })
        );
        assert_eq!(
            MacroUnitSpec::blackbox(vec![0, 1], 1, &[2], 4),
            Err(MacroUnitError::BlackboxOutputOutOfRange {
                slot: 2,
                constituents: 2
            })
        );
    }

    #[test]
    fn pinned_pyphi_macro_mapping_families_match_golden_digests() {
        let cases = [
            (
                MacroUnitSpec::coarse_grain(vec![0, 1], &[2], 4).unwrap(),
                "6d1d396bb5b5f94064365e9b44763ea3f3075352dc7d7c0a3b6a3f3d0f972821",
                "92e44a61984b014fda4028d28660004f633b29555fe3676d27fa6a0f393339f8",
                vec![false, false, false, true],
            ),
            (
                MacroUnitSpec::coarse_grain(vec![0, 1], &[1], 4).unwrap(),
                "8052c095831eaee3bb42b7ffd57f1cf63d942538dd2162f7a485f409c6b5bd7d",
                "a17a4438aba7f19ae39dcd873e81b9f6712bb6c523b71bad4ade00868e18a656",
                vec![false, true, true, false],
            ),
            (
                MacroUnitSpec::coarse_grain(vec![0, 1, 2], &[1, 2], 8).unwrap(),
                "62ade755a0ef46f57971dc503afb5f1cc93fef158b9665adae73fad761e8db52",
                "391c46f57f599fa7d8adaa3354093401c5e56f47fcc1ac077089b1534be63aab",
                vec![false, true, true, true, true, true, true, false],
            ),
            (
                MacroUnitSpec::blackbox(vec![0, 1], 2, &[0], 16).unwrap(),
                "08f99240dd532f416f4853b61a56df4bb1be44db17ed466dd4625ef78ac35a2a",
                "2c5393822d234496a9948b2bfbbd5fd681c9621e7d958aec8f344eebbe25730c",
                (0..16).map(|state| ((state >> 2) & 1) != 0).collect(),
            ),
            (
                MacroUnitSpec::blackbox(vec![0, 1], 2, &[0, 1], 16).unwrap(),
                "e055a4cd3c35ea97d152b41ef3dda99b26b0bcecddb3402592f010ef1af6f41a",
                "3b596a08be2b1d43c5ecc9a952813e1885caacd40fefbd10faea3bfb64f15339",
                (0..16).map(|state| ((state >> 2) & 0b11) == 0b11).collect(),
            ),
            (
                MacroUnitSpec::blackbox(vec![0, 1, 2], 2, &[1, 2], 64).unwrap(),
                "a591b01aecb7a615d38e275299e93a9f7da970b7f23fad630cb98de617888061",
                "9b55d56ff9eeb33cc2119d8b915bf6a06b7f3bf238d87a59660f6616f4223139",
                (0..64).map(|state| ((state >> 4) & 0b11) == 0b11).collect(),
            ),
        ];

        assert_eq!(cases.len(), 6);
        for (unit, expected_mapping_sha, expected_spec_sha, expected_mapping) in cases {
            assert_eq!(unit.mapping_sha256(), expected_mapping_sha);
            assert_eq!(unit.spec_sha256(), expected_spec_sha);
            assert!(!unit.complement_flipped());
            assert_eq!(
                (0..unit.mapping_entries())
                    .map(|state| unit.map_sequence_state(state).unwrap())
                    .collect::<Vec<_>>(),
                expected_mapping
            );
        }
    }

    #[test]
    fn background_apportionment_is_canonical_disjoint_and_digest_bound() {
        let mapping = vec![false, false, true, true];
        let base = MacroUnitSpec::new(vec![0, 1], 1, mapping.clone(), 4).unwrap();
        let apportioned = MacroUnitSpec::new_with_background(
            vec![0, 1],
            1,
            mapping.clone(),
            vec![7, 5],
            4,
        )
        .unwrap();
        let reordered = MacroUnitSpec::new_with_background(
            vec![0, 1],
            1,
            mapping,
            vec![5, 7],
            4,
        )
        .unwrap();

        assert_eq!(apportioned.background_apportionment(), &[5, 7]);
        assert_eq!(apportioned.mapping_sha256(), base.mapping_sha256());
        assert_eq!(
            apportioned.spec_sha256(),
            "d8c188273ab27e44b4f4aac4a1dee6fea99352b89011d4e643f23d33a13d12f6"
        );
        assert_ne!(apportioned.spec_sha256(), base.spec_sha256());
        assert_eq!(apportioned.spec_sha256(), reordered.spec_sha256());
        assert!(matches!(
            MacroUnitSpec::new_with_background(
                vec![0, 1], 1, vec![false, false, true, true], vec![5, 5], 4
            ),
            Err(MacroUnitError::DuplicateBackgroundApportionment { unit: 5 })
        ));
        assert!(matches!(
            MacroUnitSpec::new_with_background(
                vec![0, 1], 1, vec![false, false, true, true], vec![1], 4
            ),
            Err(MacroUnitError::BackgroundOverlapsConstituents { unit: 1 })
        ));
    }

    #[test]
    fn mapping_digest_binds_constituents_and_temporal_grain() {
        let mapping = vec![false, false, true, true];
        let base = MacroUnitSpec::new(vec![0, 1], 1, mapping.clone(), 4).unwrap();
        let other_constituents = MacroUnitSpec::new(vec![0, 2], 1, mapping.clone(), 4).unwrap();
        let other_grain = MacroUnitSpec::new(
            vec![0, 1],
            2,
            vec![
                false, false, false, false, false, false, false, true,
                false, false, false, false, false, false, false, true,
            ],
            16,
        )
        .unwrap();
        assert_ne!(base.mapping_sha256(), other_constituents.mapping_sha256());
        assert_ne!(base.mapping_sha256(), other_grain.mapping_sha256());
    }
}
