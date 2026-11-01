// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, resource-bounded representation of an IIT 4.0 macro unit.
//!
//! This module does not yet construct a macro TPM or judge the intrinsic-unit
//! postulates. It freezes the identity and truth-table semantics that those
//! later computations must consume.

use sha2::{Digest, Sha256};
use thiserror::Error;

const MACRO_MAPPING_MAGIC: &[u8] = b"SYMTHAEA-IIT-MACRO-MAPPING\0v0.1\0";
const MACRO_UNIT_SPEC_MAGIC: &[u8] = b"SYMTHAEA-IIT-MACRO-UNIT-SPEC\0v0.1\0";

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
    #[error("constant mappings do not define a non-trivial macro unit")]
    ConstantMapping,
    #[error("sequence-state index {index} is outside the mapping domain of {states} states")]
    SequenceStateOutOfRange { index: usize, states: usize },
}

/// A binary macro unit with an explicit constituent order, temporal grain,
/// truth table, and complement orientation.
///
/// The truth table's input index is a little-endian bit index. Bit
/// (time * constituents.len() + constituent_slot) selects the named
/// constituent at that time step; time is oldest-first and the first
/// constituent is the least-significant bit of the oldest micro state.
///
/// A mapping and its Boolean complement describe the same unoriented macro
/// unit with opposite macro-state labels. mapping_sha256 therefore binds the
/// canonical representative (the lexicographically smaller of the mapping and
/// its complement), while spec_sha256 additionally binds the orientation.
/// This keeps unit-shape equivalence separate from the state-label transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroUnitSpec {
    constituents: Vec<usize>,
    update_grain: usize,
    sequence_state_bits: usize,
    canonical_mapping: Vec<bool>,
    complement_flipped: bool,
    mapping_sha256: String,
    spec_sha256: String,
}

impl MacroUnitSpec {
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
        Self::new(constituents, 1, mapping, max_mapping_entries)
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
        Self::new(constituents, update_grain, mapping, max_mapping_entries)
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
        let spec_sha256 = Self::spec_digest(&mapping_sha256, complement_flipped);

        Ok(Self {
            constituents,
            update_grain,
            sequence_state_bits,
            canonical_mapping,
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

    fn spec_digest(mapping_sha256: &str, complement_flipped: bool) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MACRO_UNIT_SPEC_MAGIC);
        bytes.extend_from_slice(mapping_sha256.as_bytes());
        bytes.push(u8::from(complement_flipped));
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
