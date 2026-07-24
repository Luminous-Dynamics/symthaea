// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared block-code capability metadata.
//!
//! These values describe algebraic guarantees, not observed simulation rates.
//! In particular, the unknown-error radius is `floor((d_min - 1) / 2)` and the
//! known-erasure radius is `d_min - 1`.

/// Alphabet used by one code symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Binary symbols in `{0, 1}`.
    Bit,
    /// Full byte symbols in GF(2⁸).
    Byte,
}

/// Named code family for diagnostics and evidence manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeFamily {
    Hamming74,
    Hamming84Secded,
    Repetition,
    ReedSolomon,
}

/// Algebraic parameters for one fixed-size block code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCodeParameters {
    pub family: CodeFamily,
    pub symbol_kind: SymbolKind,
    pub message_symbols: usize,
    pub parity_symbols: usize,
    pub codeword_symbols: usize,
    pub minimum_distance: usize,
    pub unknown_error_correction_radius: usize,
    pub known_erasure_correction_radius: usize,
}

impl BlockCodeParameters {
    /// Code rate `k / n` for reporting.
    #[must_use]
    pub fn rate(self) -> f64 {
        if self.codeword_symbols == 0 {
            return 0.0;
        }
        self.message_symbols as f64 / self.codeword_symbols as f64
    }

    /// Whether an `(unknown_errors, known_erasures)` pair lies within the
    /// standard minimum-distance guarantee `2e + s < d_min`.
    #[must_use]
    pub fn supports_errata(self, unknown_errors: usize, known_erasures: usize) -> bool {
        unknown_errors
            .saturating_mul(2)
            .saturating_add(known_erasures)
            < self.minimum_distance
    }
}

pub const HAMMING74_PARAMETERS: BlockCodeParameters = BlockCodeParameters {
    family: CodeFamily::Hamming74,
    symbol_kind: SymbolKind::Bit,
    message_symbols: 4,
    parity_symbols: 3,
    codeword_symbols: 7,
    minimum_distance: 3,
    unknown_error_correction_radius: 1,
    known_erasure_correction_radius: 2,
};

pub const HAMMING84_PARAMETERS: BlockCodeParameters = BlockCodeParameters {
    family: CodeFamily::Hamming84Secded,
    symbol_kind: SymbolKind::Bit,
    message_symbols: 4,
    parity_symbols: 4,
    codeword_symbols: 8,
    minimum_distance: 4,
    unknown_error_correction_radius: 1,
    known_erasure_correction_radius: 3,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_capabilities_follow_minimum_distance() {
        assert!(HAMMING74_PARAMETERS.supports_errata(1, 0));
        assert!(HAMMING74_PARAMETERS.supports_errata(0, 2));
        assert!(!HAMMING74_PARAMETERS.supports_errata(1, 1));

        assert!(HAMMING84_PARAMETERS.supports_errata(1, 1));
        assert!(HAMMING84_PARAMETERS.supports_errata(0, 3));
        assert!(!HAMMING84_PARAMETERS.supports_errata(2, 0));
        assert_eq!(HAMMING84_PARAMETERS.rate(), 0.5);
    }
}
