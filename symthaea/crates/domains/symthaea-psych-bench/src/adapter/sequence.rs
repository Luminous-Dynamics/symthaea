// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sequence-to-HDC encoding via permutation-position binding.
//!
//! Items in a sequence are encoded as `item_hv.permute(position)` then bundled,
//! preserving both identity and order — HDC's native position encoding.

use super::StimulusAdapter;
use symthaea_core::hdc::ContinuousHV;

/// A sequence item identified by a u64 token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceItem(pub u64);

/// Encodes sequences using permutation-position binding.
///
/// Each item gets a deterministic basis vector from its token,
/// then is permuted by its position index before bundling.
pub struct SequenceAdapter;

impl SequenceAdapter {
    /// Create a deterministic basis vector for a token.
    pub fn basis_hv(token: u64, dim: usize) -> ContinuousHV {
        ContinuousHV::random(dim, token.wrapping_add(1))
    }

    /// Encode a sequence with position binding, returning the bundled result.
    pub fn encode_sequence_bundled(&self, items: &[SequenceItem], dim: usize) -> ContinuousHV {
        if items.is_empty() {
            return ContinuousHV::zero(dim);
        }
        let positioned: Vec<ContinuousHV> = items
            .iter()
            .enumerate()
            .map(|(pos, item)| {
                let basis = Self::basis_hv(item.0, dim);
                basis.permute(pos)
            })
            .collect();
        ContinuousHV::bundle_owned(&positioned)
    }
}

impl StimulusAdapter for SequenceAdapter {
    type Stimulus = SequenceItem;

    fn encode(&self, stimulus: &SequenceItem, dim: usize) -> ContinuousHV {
        Self::basis_hv(stimulus.0, dim)
    }

    fn encode_sequence(&self, stimuli: &[SequenceItem], dim: usize) -> Vec<ContinuousHV> {
        stimuli
            .iter()
            .enumerate()
            .map(|(pos, item)| {
                let basis = Self::basis_hv(item.0, dim);
                basis.permute(pos)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_deterministic() {
        let a = SequenceAdapter::basis_hv(42, 512);
        let b = SequenceAdapter::basis_hv(42, 512);
        assert!((a.similarity(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_tokens_dissimilar() {
        let a = SequenceAdapter::basis_hv(1, 512);
        let b = SequenceAdapter::basis_hv(2, 512);
        assert!(a.similarity(&b).abs() < 0.2);
    }

    #[test]
    fn test_position_matters() {
        let adapter = SequenceAdapter;
        let items = vec![SequenceItem(1), SequenceItem(2)];
        let rev_items = vec![SequenceItem(2), SequenceItem(1)];
        let a = adapter.encode_sequence_bundled(&items, 512);
        let b = adapter.encode_sequence_bundled(&rev_items, 512);
        // Different order should produce different bundles
        assert!(a.similarity(&b) < 0.9);
    }

    #[test]
    fn test_empty_sequence() {
        let adapter = SequenceAdapter;
        let result = adapter.encode_sequence_bundled(&[], 512);
        assert_eq!(result.values.len(), 512);
    }
}
