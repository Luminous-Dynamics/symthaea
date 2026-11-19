//! Deterministic predictive prefix mapping from Rabbit's four-branch flag strategy.
//!
//! This module implements only the paper's unambiguous mapping primitive.
//! It is not a full Rabbit or ACTF encoder: branch payload semantics and TOE
//! are kept separate until the paper's pseudocode ambiguities are resolved.

use std::fmt;

pub const RABBIT_BRANCH_COUNT: usize = 4;
const DYNAMIC_CODES: [RabbitFlagCode; RABBIT_BRANCH_COUNT] = [
    RabbitFlagCode { bits: 0b0, bit_len: 1 },
    RabbitFlagCode { bits: 0b10, bit_len: 2 },
    RabbitFlagCode { bits: 0b110, bit_len: 3 },
    RabbitFlagCode { bits: 0b111, bit_len: 3 },
];

/// A prefix code stored right-aligned in the low bit_len bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RabbitFlagCode {
    pub bits: u8,
    pub bit_len: u8,
}

/// A deterministic branch-to-prefix mapping for the four Rabbit branch IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RabbitPrefixMap {
    /// Branch IDs sorted by predicted frequency, highest first.
    ranked_branches: [u8; RABBIT_BRANCH_COUNT],
    /// Whether the {1,2,3,3}-bit dynamic prefix assignment is active.
    dynamic: bool,
}

impl RabbitPrefixMap {
    /// Rabbit's first-block default: identity rank order with dynamic prefix lengths.
    pub const fn first_block_default() -> Self {
        Self {
            ranked_branches: [0, 1, 2, 3],
            dynamic: true,
        }
    }

    /// Build a dynamic prefix map using previous-block branch counts.
    ///
    /// Ties are broken by ascending branch ID. The code lengths and codewords
    /// are the paper's {1,2,3,3}-bit mapping: 0, 10, 110, 111 by rank.
    pub fn from_previous_counts(counts: [u64; RABBIT_BRANCH_COUNT]) -> Self {
        let mut ranked = [0u8, 1, 2, 3];
        ranked.sort_by(|left, right| {
            counts[*right as usize]
                .cmp(&counts[*left as usize])
                .then_with(|| left.cmp(right))
        });
        Self {
            ranked_branches: ranked,
            dynamic: true,
        }
    }

    /// Build the dynamic map only when its predicted expected flag cost beats
    /// fixed two-bit branch IDs, using the previous block's sorted frequencies.
    ///
    /// The paper derives the profitability condition x1 > x3 + x4. This gate
    /// is exposed as an explicit experimental policy because Algorithm 1 does
    /// not clearly show where the paper applies the gate.
    pub fn gated_by_previous_counts(
        counts: [u64; RABBIT_BRANCH_COUNT],
    ) -> Self {
        let dynamic = Self::from_previous_counts(counts);
        if dynamic.predicted_gain_positive(counts) {
            dynamic
        } else {
            Self {
                ranked_branches: [0, 1, 2, 3],
                dynamic: false,
            }
        }
    }

    pub const fn is_dynamic(self) -> bool {
        self.dynamic
    }

    pub const fn ranked_branches(self) -> [u8; RABBIT_BRANCH_COUNT] {
        self.ranked_branches
    }

    /// Return the flag code for a semantic branch ID.
    pub fn code_for_branch(self, branch: u8) -> Result<RabbitFlagCode, RabbitMappingError> {
        let branch_index = usize::from(branch);
        if branch_index >= RABBIT_BRANCH_COUNT {
            return Err(RabbitMappingError::InvalidBranch(branch));
        }
        if !self.dynamic {
            return Ok(RabbitFlagCode {
                bits: branch,
                bit_len: 2,
            });
        }
        let rank = self
            .ranked_branches
            .iter()
            .position(|candidate| *candidate == branch)
            .ok_or(RabbitMappingError::InvalidMapping)?;
        Ok(DYNAMIC_CODES[rank])
    }

    /// Decode a right-aligned prefix from the first available_bits bits.
    ///
    /// The prefix is read most-significant-bit first. Returns the semantic
    /// branch ID and the number of consumed bits; None means insufficient bits.
    pub fn decode_prefix(
        self,
        prefix: u8,
        available_bits: u8,
    ) -> Result<Option<(u8, u8)>, RabbitMappingError> {
        if !self.dynamic {
            if available_bits < 2 {
                return Ok(None);
            }
            let branch = (prefix >> (available_bits - 2)) & 0b11;
            return Ok(Some((branch, 2)));
        }

        if available_bits == 0 {
            return Ok(None);
        }
        let first = (prefix >> (available_bits - 1)) & 1;
        if first == 0 {
            return Ok(Some((self.ranked_branches[0], 1)));
        }
        if available_bits < 2 {
            return Ok(None);
        }
        let second = (prefix >> (available_bits - 2)) & 1;
        if second == 0 {
            return Ok(Some((self.ranked_branches[1], 2)));
        }
        if available_bits < 3 {
            return Ok(None);
        }
        let third = (prefix >> (available_bits - 3)) & 1;
        if third == 0 {
            Ok(Some((self.ranked_branches[2], 3)))
        } else {
            Ok(Some((self.ranked_branches[3], 3)))
        }
    }

    /// Whether the previous-block prediction meets the paper's theoretical
    /// condition x1 > x3 + x4 for an improvement over fixed two-bit flags.
    pub fn predicted_gain_positive(self, counts: [u64; RABBIT_BRANCH_COUNT]) -> bool {
        let mut sorted = counts;
        sorted.sort_unstable_by(|left, right| right.cmp(left));
        u128::from(sorted[0]) > u128::from(sorted[2]) + u128::from(sorted[3])
    }
}

/// Invalid branch IDs or malformed mapping state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RabbitMappingError {
    InvalidBranch(u8),
    InvalidMapping,
}

impl fmt::Display for RabbitMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBranch(branch) => write!(f, "invalid Rabbit branch ID {branch}"),
            Self::InvalidMapping => write!(f, "invalid Rabbit branch rank mapping"),
        }
    }
}

impl std::error::Error for RabbitMappingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_block_uses_paper_default_prefix_order() {
        let mapping = RabbitPrefixMap::first_block_default();
        assert_eq!(mapping.code_for_branch(0).unwrap(), RabbitFlagCode { bits: 0, bit_len: 1 });
        assert_eq!(mapping.code_for_branch(1).unwrap(), RabbitFlagCode { bits: 0b10, bit_len: 2 });
        assert_eq!(mapping.code_for_branch(2).unwrap(), RabbitFlagCode { bits: 0b110, bit_len: 3 });
        assert_eq!(mapping.code_for_branch(3).unwrap(), RabbitFlagCode { bits: 0b111, bit_len: 3 });
    }

    #[test]
    fn previous_block_counts_assign_codes_by_descending_frequency() {
        let mapping = RabbitPrefixMap::from_previous_counts([10, 40, 30, 20]);
        assert_eq!(mapping.ranked_branches(), [1, 2, 3, 0]);
        assert_eq!(mapping.code_for_branch(1).unwrap(), DYNAMIC_CODES[0]);
        assert_eq!(mapping.code_for_branch(2).unwrap(), DYNAMIC_CODES[1]);
        assert_eq!(mapping.code_for_branch(3).unwrap(), DYNAMIC_CODES[2]);
        assert_eq!(mapping.code_for_branch(0).unwrap(), DYNAMIC_CODES[3]);
    }

    #[test]
    fn frequency_ties_break_by_branch_id() {
        let mapping = RabbitPrefixMap::from_previous_counts([5, 5, 7, 1]);
        assert_eq!(mapping.ranked_branches(), [2, 0, 1, 3]);
    }

    #[test]
    fn prefix_decoder_round_trips_all_dynamic_branches() {
        for mapping in [
            RabbitPrefixMap::first_block_default(),
            RabbitPrefixMap::from_previous_counts([10, 40, 30, 20]),
        ] {
            for branch in 0..RABBIT_BRANCH_COUNT as u8 {
                let code = mapping.code_for_branch(branch).unwrap();
                assert_eq!(
                    mapping.decode_prefix(code.bits, code.bit_len).unwrap(),
                    Some((branch, code.bit_len))
                );
            }
        }
    }

    #[test]
    fn dynamic_gain_condition_matches_paper_inequality() {
        assert!(RabbitPrefixMap::first_block_default().predicted_gain_positive([40, 30, 20, 10]));
        assert!(!RabbitPrefixMap::first_block_default().predicted_gain_positive([25, 25, 25, 25]));
        assert!(!RabbitPrefixMap::first_block_default().predicted_gain_positive([30, 30, 25, 15]));
    }

    #[test]
    fn optional_profitability_gate_falls_back_to_two_bit_ids() {
        let dynamic = RabbitPrefixMap::gated_by_previous_counts([40, 30, 20, 10]);
        assert!(dynamic.is_dynamic());
        let fixed = RabbitPrefixMap::gated_by_previous_counts([25, 25, 25, 25]);
        assert!(!fixed.is_dynamic());
        assert_eq!(fixed.code_for_branch(3).unwrap(), RabbitFlagCode { bits: 3, bit_len: 2 });
    }

    #[test]
    fn invalid_branch_is_rejected() {
        assert_eq!(
            RabbitPrefixMap::first_block_default().code_for_branch(4),
            Err(RabbitMappingError::InvalidBranch(4))
        );
    }
}
