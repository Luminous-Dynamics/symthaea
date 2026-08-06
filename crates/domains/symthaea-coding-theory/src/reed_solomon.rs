// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reed-Solomon codes over **GF(2⁸)** using the AES irreducible polynomial
//! supplied by `symthaea-finite-field`.
//!
//! The AES field polynomial alone does not choose a generator for the
//! multiplicative group. In this field `0x02` has order 51, while `0x03` is a
//! primitive element of order 255. This module therefore makes the primitive
//! element and first consecutive root explicit and validates them before use.
//!
//! Polynomials are represented most-significant coefficient first. The default
//! generator has roots `α⁰ … α^{nsym−1}` with `α = 0x03`.

use std::fmt;

use symthaea_finite_field::binary::{add, mul, pow};

use crate::parameters::{BlockCodeParameters, CodeFamily, SymbolKind};

/// Maximum number of symbols in a Reed-Solomon codeword over GF(2⁸).
pub const MAX_CODEWORD_LEN: usize = 255;

/// AES irreducible polynomial `x^8 + x^4 + x^3 + x + 1`.
pub const AES_FIELD_POLYNOMIAL: u16 = 0x11B;

/// A primitive element of the AES field GF(2⁸) / `(x⁸ + x⁴ + x³ + x + 1)`.
pub const AES_PRIMITIVE_ELEMENT: u8 = 0x03;

/// Errors raised by checked Reed-Solomon construction and operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReedSolomonError {
    /// At least one parity symbol is required.
    ZeroParitySymbols,
    /// A non-empty message leaves room for at most 254 parity symbols.
    TooManyParitySymbols { requested: usize },
    /// Zero cannot generate the non-zero multiplicative group.
    ZeroPrimitiveElement,
    /// The configured element does not have multiplicative order 255.
    NonPrimitiveElement { element: u8 },
    /// The encoded message would exceed the 255-symbol field limit.
    MessageTooLong {
        message_len: usize,
        parity_symbols: usize,
    },
    /// A supplied word exceeds the 255-symbol field limit.
    CodewordTooLong { codeword_len: usize },
    /// A fixed frame received the wrong number of source symbols.
    MessageLengthMismatch { expected: usize, actual: usize },
    /// A fixed frame received the wrong number of codeword symbols.
    CodewordLengthMismatch { expected: usize, actual: usize },
    /// A shortened frame cannot be longer than its declared parent message.
    InvalidShortening {
        parent_message_symbols: usize,
        transmitted_message_symbols: usize,
    },
    /// A purported parent codeword has non-zero symbols in its shortened prefix.
    NonZeroShorteningPrefix { position: usize, value: u8 },
    /// A caller-owned encoding buffer had the wrong length.
    OutputLengthMismatch { expected: usize, actual: usize },
    /// A caller-owned parity buffer did not match the configured parity width.
    ParityLengthMismatch { expected: usize, actual: usize },
    /// A word does not contain enough symbols to include the configured parity.
    CodewordTooShort {
        codeword_len: usize,
        parity_symbols: usize,
    },
    /// The inferred locator degree exceeds the bounded-distance capacity.
    TooManyErrors {
        locator_degree: usize,
        correction_capacity: usize,
    },
    /// Chien search did not find the number of roots promised by the locator.
    LocatorFailure { expected: usize, found: usize },
    /// Error magnitudes could not be uniquely determined.
    SingularMagnitudeSystem,
    /// An erasure position lies outside the supplied codeword.
    InvalidErasurePosition {
        position: usize,
        codeword_len: usize,
    },
    /// The same erasure position was supplied more than once.
    DuplicateErasurePosition { position: usize },
    /// A code can recover at most one known erasure per parity symbol.
    TooManyErasures {
        erasures: usize,
        parity_symbols: usize,
    },
    /// Mixed unknown errors and known erasures exceed `2e + s <= parity`.
    TooManyErrata {
        errors: usize,
        erasures: usize,
        parity_symbols: usize,
    },
    /// An inferred unknown-error location duplicates a declared erasure.
    ErrorErasurePositionCollision { position: usize },
    /// Non-zero syndromes remain but no erasure locations were supplied.
    UnlocatedErrorsPresent,
    /// A caller policy promises a correction envelope outside the algebraic bound.
    InvalidDecodePolicy {
        max_unknown_errors: usize,
        max_known_erasures: usize,
        parity_symbols: usize,
    },
    /// A received frame declared more erasures than the caller permits.
    PolicyErasureBudgetExceeded { declared: usize, maximum: usize },
    /// Decoding required more unknown-error corrections than the caller permits.
    PolicyUnknownErrorBudgetExceeded { corrected: usize, maximum: usize },
    /// Candidate corrections did not restore every syndrome to zero.
    CorrectionVerificationFailed,
}

impl fmt::Display for ReedSolomonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroParitySymbols => {
                write!(f, "Reed-Solomon requires at least one parity symbol")
            }
            Self::TooManyParitySymbols { requested } => write!(
                f,
                "Reed-Solomon over GF(2^8) supports at most 254 parity symbols, got {requested}"
            ),
            Self::ZeroPrimitiveElement => write!(f, "primitive element must be non-zero"),
            Self::NonPrimitiveElement { element } => write!(
                f,
                "0x{element:02x} does not have multiplicative order 255 in the configured field"
            ),
            Self::MessageTooLong {
                message_len,
                parity_symbols,
            } => write!(
                f,
                "message ({message_len}) plus parity ({parity_symbols}) exceeds 255 symbols"
            ),
            Self::CodewordTooLong { codeword_len } => {
                write!(f, "codeword length {codeword_len} exceeds 255 symbols")
            }
            Self::MessageLengthMismatch { expected, actual } => write!(
                f,
                "message length {actual} does not match fixed frame length {expected}"
            ),
            Self::CodewordLengthMismatch { expected, actual } => write!(
                f,
                "codeword length {actual} does not match fixed frame length {expected}"
            ),
            Self::InvalidShortening {
                parent_message_symbols,
                transmitted_message_symbols,
            } => write!(
                f,
                "shortened message length {transmitted_message_symbols} exceeds parent message length {parent_message_symbols}"
            ),
            Self::NonZeroShorteningPrefix { position, value } => write!(
                f,
                "parent codeword shortening prefix contains 0x{value:02x} at position {position}"
            ),
            Self::OutputLengthMismatch { expected, actual } => write!(
                f,
                "output length {actual} does not match required codeword length {expected}"
            ),
            Self::ParityLengthMismatch { expected, actual } => write!(
                f,
                "parity output length {actual} does not match configured parity width {expected}"
            ),
            Self::CodewordTooShort {
                codeword_len,
                parity_symbols,
            } => write!(
                f,
                "codeword length {codeword_len} is shorter than its {parity_symbols} parity symbols"
            ),
            Self::TooManyErrors {
                locator_degree,
                correction_capacity,
            } => write!(
                f,
                "error locator degree {locator_degree} exceeds correction capacity {correction_capacity}"
            ),
            Self::LocatorFailure { expected, found } => write!(
                f,
                "error locator promised {expected} roots but Chien search found {found}"
            ),
            Self::SingularMagnitudeSystem => {
                write!(f, "error magnitudes are not uniquely solvable")
            }
            Self::InvalidErasurePosition {
                position,
                codeword_len,
            } => write!(
                f,
                "erasure position {position} is outside codeword length {codeword_len}"
            ),
            Self::DuplicateErasurePosition { position } => {
                write!(f, "erasure position {position} was supplied more than once")
            }
            Self::TooManyErasures {
                erasures,
                parity_symbols,
            } => write!(
                f,
                "{erasures} erasures exceed the {parity_symbols}-symbol erasure capacity"
            ),
            Self::TooManyErrata {
                errors,
                erasures,
                parity_symbols,
            } => write!(
                f,
                "{errors} unknown errors plus {erasures} erasures exceed the mixed capacity 2e + s <= {parity_symbols}"
            ),
            Self::ErrorErasurePositionCollision { position } => write!(
                f,
                "inferred unknown error position {position} duplicates a declared erasure"
            ),
            Self::UnlocatedErrorsPresent => write!(
                f,
                "non-zero syndromes remain but no erasure locations were supplied"
            ),
            Self::InvalidDecodePolicy {
                max_unknown_errors,
                max_known_erasures,
                parity_symbols,
            } => write!(
                f,
                "decode policy 2*{max_unknown_errors} + {max_known_erasures} exceeds {parity_symbols} parity symbols"
            ),
            Self::PolicyErasureBudgetExceeded { declared, maximum } => write!(
                f,
                "declared erasure count {declared} exceeds caller policy maximum {maximum}"
            ),
            Self::PolicyUnknownErrorBudgetExceeded { corrected, maximum } => write!(
                f,
                "decoder corrected {corrected} unknown errors, exceeding caller policy maximum {maximum}"
            ),
            Self::CorrectionVerificationFailed => {
                write!(
                    f,
                    "candidate Reed-Solomon correction failed syndrome verification"
                )
            }
        }
    }
}

impl std::error::Error for ReedSolomonError {}

/// A corrected Reed-Solomon symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolCorrection {
    /// Zero-based symbol index in the received codeword.
    pub position: usize,
    /// Non-zero field magnitude XORed into the received symbol.
    pub magnitude: u8,
}

/// Successful bounded-distance Reed-Solomon decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomonDecodeReport {
    /// Recovered systematic message without parity symbols.
    pub message: Vec<u8>,
    /// Fully corrected systematic codeword.
    pub corrected_codeword: Vec<u8>,
    /// Corrected symbol positions and magnitudes.
    pub corrections: Vec<SymbolCorrection>,
    /// Syndromes observed before correction for diagnostics and evidence.
    pub syndromes_before: Vec<u8>,
}

impl ReedSolomonDecodeReport {
    /// Number of symbols actually changed by the decoder.
    #[must_use]
    pub fn corrected_symbols(&self) -> usize {
        self.corrections.len()
    }

    /// Number of corrections that coincide with caller-declared erasures.
    #[must_use]
    pub fn corrected_erasures(&self, erasure_positions: &[usize]) -> usize {
        self.corrections
            .iter()
            .filter(|correction| erasure_positions.contains(&correction.position))
            .count()
    }

    /// Number of corrections inferred at positions not declared erased.
    #[must_use]
    pub fn corrected_unknown_errors(&self, erasure_positions: &[usize]) -> usize {
        self.corrected_symbols()
            .saturating_sub(self.corrected_erasures(erasure_positions))
    }

    /// Minimum-distance errata weight `2e + s` represented by this report.
    #[must_use]
    pub fn corrected_errata_weight(&self, erasure_positions: &[usize]) -> usize {
        self.corrected_unknown_errors(erasure_positions)
            .saturating_mul(2)
            .saturating_add(erasure_positions.len())
    }
}

/// Caller-selected fail-closed envelope for one Reed-Solomon decode.
///
/// The policy may be tighter than the code's algebraic capacity, but never
/// looser. This is useful when a protocol, storage format, or safety case has
/// independently budgeted fewer corruptions than the codec could repair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonDecodePolicy {
    pub max_unknown_errors: usize,
    pub max_known_erasures: usize,
}

impl ReedSolomonDecodePolicy {
    #[must_use]
    pub const fn new(max_unknown_errors: usize, max_known_erasures: usize) -> Self {
        Self {
            max_unknown_errors,
            max_known_erasures,
        }
    }

    /// The full unknown-error-only radius for a codec configuration.
    #[must_use]
    pub const fn unknown_errors_only(config: ReedSolomonConfig) -> Self {
        Self::new(config.parity_symbols / 2, 0)
    }

    /// Validate that the promised envelope lies within `2e + s <= nsym`.
    pub fn validate(self, config: ReedSolomonConfig) -> Result<(), ReedSolomonError> {
        if self
            .max_unknown_errors
            .saturating_mul(2)
            .saturating_add(self.max_known_erasures)
            > config.parity_symbols
        {
            return Err(ReedSolomonError::InvalidDecodePolicy {
                max_unknown_errors: self.max_unknown_errors,
                max_known_erasures: self.max_known_erasures,
                parity_symbols: config.parity_symbols,
            });
        }
        Ok(())
    }
}

/// Explicit Reed-Solomon field and root convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReedSolomonConfig {
    /// Number of parity symbols appended to each message.
    pub parity_symbols: usize,
    /// Generator of the 255 non-zero field elements.
    pub primitive_element: u8,
    /// Exponent of the first consecutive generator root.
    pub first_root: u8,
}

impl ReedSolomonConfig {
    /// The crate's interoperable default: AES field, primitive element `0x03`,
    /// roots beginning at `α⁰`.
    #[must_use]
    pub const fn aes(parity_symbols: usize) -> Self {
        Self {
            parity_symbols,
            primitive_element: AES_PRIMITIVE_ELEMENT,
            first_root: 0,
        }
    }

    /// Validate the parity count and multiplicative generator.
    pub fn validate(self) -> Result<(), ReedSolomonError> {
        if self.parity_symbols == 0 {
            return Err(ReedSolomonError::ZeroParitySymbols);
        }
        if self.parity_symbols >= MAX_CODEWORD_LEN {
            return Err(ReedSolomonError::TooManyParitySymbols {
                requested: self.parity_symbols,
            });
        }
        if self.primitive_element == 0 {
            return Err(ReedSolomonError::ZeroPrimitiveElement);
        }
        if !has_order_255(self.primitive_element) {
            return Err(ReedSolomonError::NonPrimitiveElement {
                element: self.primitive_element,
            });
        }
        Ok(())
    }
}

/// Fixed-size Reed-Solomon frame contract.
///
/// This wrapper binds a codec to one `k` and `n`, preventing accidental use of
/// the right parity profile with the wrong frame length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomonFrame {
    codec: ReedSolomon,
    message_symbols: usize,
    codeword_symbols: usize,
}

impl ReedSolomonFrame {
    /// Construct a fixed frame from a validated codec configuration.
    pub fn new(
        config: ReedSolomonConfig,
        message_symbols: usize,
    ) -> Result<Self, ReedSolomonError> {
        Self::from_codec(ReedSolomon::new(config)?, message_symbols)
    }

    /// Bind an existing codec to one exact message length.
    pub fn from_codec(
        codec: ReedSolomon,
        message_symbols: usize,
    ) -> Result<Self, ReedSolomonError> {
        let codeword_symbols = codec.encoded_len(message_symbols)?;
        Ok(Self {
            codec,
            message_symbols,
            codeword_symbols,
        })
    }

    #[must_use]
    pub const fn message_symbols(&self) -> usize {
        self.message_symbols
    }

    #[must_use]
    pub const fn codeword_symbols(&self) -> usize {
        self.codeword_symbols
    }

    #[must_use]
    pub fn codec(&self) -> &ReedSolomon {
        &self.codec
    }

    #[must_use]
    pub fn parameters(&self) -> BlockCodeParameters {
        self.codec
            .parameters(self.message_symbols)
            .expect("frame construction already validated its message length")
    }

    pub fn encode(&self, message: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        self.validate_message(message)?;
        self.codec.encode(message)
    }

    pub fn encode_into(&self, message: &[u8], codeword: &mut [u8]) -> Result<(), ReedSolomonError> {
        self.validate_message(message)?;
        self.validate_codeword(codeword)?;
        self.codec.encode_into(message, codeword)
    }

    pub fn decode(&self, codeword: &[u8]) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword(codeword)?;
        self.codec.decode(codeword)
    }

    pub fn decode_erasures(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword(codeword)?;
        self.codec.decode_erasures(codeword, erasure_positions)
    }

    pub fn decode_with_erasures(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword(codeword)?;
        self.codec.decode_with_erasures(codeword, erasure_positions)
    }

    /// Decode under a caller-selected correction envelope.
    pub fn decode_with_policy(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
        policy: ReedSolomonDecodePolicy,
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword(codeword)?;
        self.codec
            .decode_with_policy(codeword, erasure_positions, policy)
    }

    fn validate_message(&self, message: &[u8]) -> Result<(), ReedSolomonError> {
        if message.len() != self.message_symbols {
            return Err(ReedSolomonError::MessageLengthMismatch {
                expected: self.message_symbols,
                actual: message.len(),
            });
        }
        Ok(())
    }

    fn validate_codeword(&self, codeword: &[u8]) -> Result<(), ReedSolomonError> {
        if codeword.len() != self.codeword_symbols {
            return Err(ReedSolomonError::CodewordLengthMismatch {
                expected: self.codeword_symbols,
                actual: codeword.len(),
            });
        }
        Ok(())
    }
}

/// Explicit shortened Reed-Solomon frame.
///
/// Shortening removes a known all-zero prefix from a longer parent message and
/// its systematic codeword. Because the encoder starts from the all-zero
/// remainder, encoding the transmitted suffix directly is identical to
/// encoding `zero_prefix || message` and removing the same prefix. This type
/// records that parent contract for interoperability and evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomonShortenedFrame {
    frame: ReedSolomonFrame,
    parent_message_symbols: usize,
    parent_codeword_symbols: usize,
    shortening_symbols: usize,
}

impl ReedSolomonShortenedFrame {
    /// Bind one transmitted `k` to a larger canonical parent `k_parent`.
    pub fn new(
        config: ReedSolomonConfig,
        parent_message_symbols: usize,
        transmitted_message_symbols: usize,
    ) -> Result<Self, ReedSolomonError> {
        if transmitted_message_symbols > parent_message_symbols {
            return Err(ReedSolomonError::InvalidShortening {
                parent_message_symbols,
                transmitted_message_symbols,
            });
        }
        let codec = ReedSolomon::new(config)?;
        let parent_codeword_symbols = codec.encoded_len(parent_message_symbols)?;
        let frame = ReedSolomonFrame::from_codec(codec, transmitted_message_symbols)?;
        Ok(Self {
            frame,
            parent_message_symbols,
            parent_codeword_symbols,
            shortening_symbols: parent_message_symbols - transmitted_message_symbols,
        })
    }

    #[must_use]
    pub const fn parent_message_symbols(&self) -> usize {
        self.parent_message_symbols
    }

    #[must_use]
    pub const fn parent_codeword_symbols(&self) -> usize {
        self.parent_codeword_symbols
    }

    #[must_use]
    pub const fn transmitted_message_symbols(&self) -> usize {
        self.frame.message_symbols
    }

    #[must_use]
    pub const fn transmitted_codeword_symbols(&self) -> usize {
        self.frame.codeword_symbols
    }

    #[must_use]
    pub const fn shortening_symbols(&self) -> usize {
        self.shortening_symbols
    }

    #[must_use]
    pub fn frame(&self) -> &ReedSolomonFrame {
        &self.frame
    }

    #[must_use]
    pub fn parameters(&self) -> BlockCodeParameters {
        self.frame.parameters()
    }

    #[must_use]
    pub fn parent_parameters(&self) -> BlockCodeParameters {
        self.frame
            .codec
            .parameters(self.parent_message_symbols)
            .expect("shortened frame construction validated the parent length")
    }

    pub fn encode(&self, message: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        self.frame.encode(message)
    }

    pub fn decode(&self, codeword: &[u8]) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.frame.decode(codeword)
    }

    pub fn decode_with_erasures(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.frame.decode_with_erasures(codeword, erasure_positions)
    }

    pub fn decode_with_policy(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
        policy: ReedSolomonDecodePolicy,
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.frame
            .decode_with_policy(codeword, erasure_positions, policy)
    }

    /// Reconstruct the canonical parent message by restoring its zero prefix.
    pub fn expand_message(&self, message: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        self.frame.validate_message(message)?;
        let mut parent = vec![0; self.parent_message_symbols];
        parent[self.shortening_symbols..].copy_from_slice(message);
        Ok(parent)
    }

    /// Reconstruct the canonical parent codeword by restoring its zero prefix.
    pub fn expand_codeword(&self, codeword: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        self.frame.validate_codeword(codeword)?;
        let mut parent = vec![0; self.parent_codeword_symbols];
        parent[self.shortening_symbols..].copy_from_slice(codeword);
        Ok(parent)
    }

    /// Remove the known zero prefix from a canonical parent codeword.
    pub fn contract_parent_codeword(
        &self,
        parent_codeword: &[u8],
    ) -> Result<Vec<u8>, ReedSolomonError> {
        if parent_codeword.len() != self.parent_codeword_symbols {
            return Err(ReedSolomonError::CodewordLengthMismatch {
                expected: self.parent_codeword_symbols,
                actual: parent_codeword.len(),
            });
        }
        for (position, &value) in parent_codeword[..self.shortening_symbols]
            .iter()
            .enumerate()
        {
            if value != 0 {
                return Err(ReedSolomonError::NonZeroShorteningPrefix { position, value });
            }
        }
        Ok(parent_codeword[self.shortening_symbols..].to_vec())
    }
}

/// Incremental parity accumulator for chunked Reed-Solomon messages.
///
/// The state retains only `parity_symbols` bytes. It can therefore compute the
/// systematic parity tail while a message is read from a stream, file, or
/// packet sequence. The caller remains responsible for retaining or forwarding
/// the systematic message bytes themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomonParityState<'a> {
    codec: &'a ReedSolomon,
    parity: Vec<u8>,
    message_symbols: usize,
}

impl ReedSolomonParityState<'_> {
    /// Add one chunk to the parity calculation.
    ///
    /// Length validation happens before mutation, so an overlong chunk leaves
    /// the accumulator unchanged.
    pub fn update(&mut self, symbols: &[u8]) -> Result<(), ReedSolomonError> {
        let next_len = self.message_symbols.checked_add(symbols.len()).ok_or(
            ReedSolomonError::MessageTooLong {
                message_len: usize::MAX,
                parity_symbols: self.codec.config.parity_symbols,
            },
        )?;
        self.codec.encoded_len(next_len)?;

        for &symbol in symbols {
            let feedback = add(symbol, self.parity[0]);
            self.parity.copy_within(1.., 0);
            let last = self.parity.len() - 1;
            self.parity[last] = 0;
            if feedback != 0 {
                for (parity, &coefficient) in self.parity.iter_mut().zip(&self.codec.generator[1..])
                {
                    *parity = add(*parity, mul(coefficient, feedback));
                }
            }
        }
        self.message_symbols = next_len;
        Ok(())
    }

    /// Add one source symbol.
    pub fn update_symbol(&mut self, symbol: u8) -> Result<(), ReedSolomonError> {
        self.update(&[symbol])
    }

    /// Number of source symbols incorporated so far.
    #[must_use]
    pub const fn message_symbols(&self) -> usize {
        self.message_symbols
    }

    /// Current parity remainder.
    #[must_use]
    pub fn parity(&self) -> &[u8] {
        &self.parity
    }

    /// Reset the accumulator for another message with the same codec.
    pub fn reset(&mut self) {
        self.parity.fill(0);
        self.message_symbols = 0;
    }

    /// Consume the state and return the parity tail.
    #[must_use]
    pub fn finalize(self) -> Vec<u8> {
        self.parity
    }
}

/// Reusable encoder and syndrome checker with a cached generator polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReedSolomon {
    config: ReedSolomonConfig,
    generator: Vec<u8>,
}

impl ReedSolomon {
    /// Construct and validate a codec, caching its generator polynomial.
    pub fn new(config: ReedSolomonConfig) -> Result<Self, ReedSolomonError> {
        config.validate()?;
        let generator = build_generator(config);
        Ok(Self { config, generator })
    }

    /// Return the validated field and root convention.
    #[must_use]
    pub const fn config(&self) -> ReedSolomonConfig {
        self.config
    }

    /// Algebraic parameters for one message length.
    pub fn parameters(
        &self,
        message_symbols: usize,
    ) -> Result<BlockCodeParameters, ReedSolomonError> {
        let codeword_symbols = self.encoded_len(message_symbols)?;
        let minimum_distance = self.config.parity_symbols + 1;
        Ok(BlockCodeParameters {
            family: CodeFamily::ReedSolomon,
            symbol_kind: SymbolKind::Byte,
            message_symbols,
            parity_symbols: self.config.parity_symbols,
            codeword_symbols,
            minimum_distance,
            unknown_error_correction_radius: self.config.parity_symbols / 2,
            known_erasure_correction_radius: self.config.parity_symbols,
        })
    }

    /// Return the cached monic generator polynomial.
    #[must_use]
    pub fn generator(&self) -> &[u8] {
        &self.generator
    }

    /// Required systematic codeword length for `message_len` source symbols.
    pub fn encoded_len(&self, message_len: usize) -> Result<usize, ReedSolomonError> {
        let codeword_len = message_len.checked_add(self.config.parity_symbols).ok_or(
            ReedSolomonError::MessageTooLong {
                message_len,
                parity_symbols: self.config.parity_symbols,
            },
        )?;
        if codeword_len > MAX_CODEWORD_LEN {
            return Err(ReedSolomonError::MessageTooLong {
                message_len,
                parity_symbols: self.config.parity_symbols,
            });
        }
        Ok(codeword_len)
    }

    /// Start an incremental parity calculation.
    #[must_use]
    pub fn parity_state(&self) -> ReedSolomonParityState<'_> {
        ReedSolomonParityState {
            codec: self,
            parity: vec![0; self.config.parity_symbols],
            message_symbols: 0,
        }
    }

    /// Compute only the parity tail for a systematic message.
    pub fn encode_parity(&self, message: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        let mut parity = vec![0; self.config.parity_symbols];
        self.encode_parity_into(message, &mut parity)?;
        Ok(parity)
    }

    /// Compute parity into an exactly-sized caller-owned buffer.
    pub fn encode_parity_into(
        &self,
        message: &[u8],
        parity: &mut [u8],
    ) -> Result<(), ReedSolomonError> {
        if parity.len() != self.config.parity_symbols {
            return Err(ReedSolomonError::ParityLengthMismatch {
                expected: self.config.parity_symbols,
                actual: parity.len(),
            });
        }
        self.encoded_len(message.len())?;
        let mut state = self.parity_state();
        state.update(message)?;
        parity.copy_from_slice(state.parity());
        Ok(())
    }

    /// Systematically encode `message` as `message ++ parity`.
    pub fn encode(&self, message: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        let mut codeword = vec![0; self.encoded_len(message.len())?];
        self.encode_into(message, &mut codeword)?;
        Ok(codeword)
    }

    /// Encode into an exactly-sized caller-owned codeword buffer.
    ///
    /// This avoids the temporary work allocation used by simpler encoders.
    /// The buffer is used for synthetic division and the systematic message
    /// prefix is restored before return.
    pub fn encode_into(&self, message: &[u8], codeword: &mut [u8]) -> Result<(), ReedSolomonError> {
        let expected = self.encoded_len(message.len())?;
        if codeword.len() != expected {
            return Err(ReedSolomonError::OutputLengthMismatch {
                expected,
                actual: codeword.len(),
            });
        }

        codeword[..message.len()].copy_from_slice(message);
        codeword[message.len()..].fill(0);

        for i in 0..message.len() {
            let coefficient = codeword[i];
            if coefficient != 0 {
                for j in 1..self.generator.len() {
                    codeword[i + j] = add(codeword[i + j], mul(self.generator[j], coefficient));
                }
            }
        }

        codeword[..message.len()].copy_from_slice(message);
        Ok(())
    }

    /// Evaluate a word at every configured generator root.
    pub fn syndromes(&self, codeword: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
        if codeword.len() > MAX_CODEWORD_LEN {
            return Err(ReedSolomonError::CodewordTooLong {
                codeword_len: codeword.len(),
            });
        }
        Ok((0..self.config.parity_symbols)
            .map(|i| poly_eval(codeword, self.root(i)))
            .collect())
    }

    /// Return whether every configured syndrome is zero.
    pub fn is_valid(&self, codeword: &[u8]) -> Result<bool, ReedSolomonError> {
        if codeword.len() < self.config.parity_symbols {
            return Err(ReedSolomonError::CodewordTooShort {
                codeword_len: codeword.len(),
                parity_symbols: self.config.parity_symbols,
            });
        }
        Ok(self
            .syndromes(codeword)?
            .iter()
            .all(|&syndrome| syndrome == 0))
    }

    /// Recover up to `parity_symbols` symbol erasures when their positions are known.
    ///
    /// The received bytes at erasure positions may contain any placeholder value.
    /// Magnitudes are solved directly from the leading syndromes, then every
    /// configured syndrome is recomputed before data is returned. This method is
    /// deliberately erasure-only: unknown errors outside `erasure_positions`
    /// cause post-correction verification to fail.
    pub fn decode_erasures(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword_len(codeword)?;
        let positions = self.validate_erasure_positions(codeword.len(), erasure_positions)?;
        let syndromes_before = self.syndromes(codeword)?;

        if syndromes_before.iter().all(|&syndrome| syndrome == 0) {
            return Ok(self.clean_report(codeword, syndromes_before));
        }
        if positions.is_empty() {
            return Err(ReedSolomonError::UnlocatedErrorsPresent);
        }

        let magnitudes = self.solve_error_magnitudes(
            &positions,
            codeword.len(),
            &syndromes_before[..positions.len()],
        )?;
        self.correct_and_verify(codeword, syndromes_before, &positions, &magnitudes)
    }

    /// Correct a mixture of unknown symbol errors and known erasures.
    ///
    /// Recovery is guaranteed within the Reed-Solomon mixed-errata bound
    /// `2 * unknown_errors + erasures <= parity_symbols`. Declared erasure
    /// positions may contain arbitrary placeholder bytes. The decoder removes
    /// their known locator factors from the syndrome sequence, infers only the
    /// remaining unknown-error locator, solves all magnitudes exactly, and
    /// releases data only after every configured syndrome is zero.
    pub fn decode_with_erasures(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword_len(codeword)?;
        let erasures = self.validate_erasure_positions(codeword.len(), erasure_positions)?;
        let syndromes_before = self.syndromes(codeword)?;

        if syndromes_before.iter().all(|&syndrome| syndrome == 0) {
            return Ok(self.clean_report(codeword, syndromes_before));
        }

        let modified_syndromes =
            self.remove_erasure_factors(&syndromes_before, &erasures, codeword.len());
        let unknown_locator = berlekamp_massey(&modified_syndromes)?;
        let unknown_errors = unknown_locator.len().saturating_sub(1);

        if unknown_errors
            .saturating_mul(2)
            .saturating_add(erasures.len())
            > self.config.parity_symbols
        {
            return Err(ReedSolomonError::TooManyErrata {
                errors: unknown_errors,
                erasures: erasures.len(),
                parity_symbols: self.config.parity_symbols,
            });
        }

        let unknown_positions = self.find_error_positions(&unknown_locator, codeword.len());
        if unknown_positions.len() != unknown_errors {
            return Err(ReedSolomonError::LocatorFailure {
                expected: unknown_errors,
                found: unknown_positions.len(),
            });
        }

        for &position in &unknown_positions {
            if erasures.binary_search(&position).is_ok() {
                return Err(ReedSolomonError::ErrorErasurePositionCollision { position });
            }
        }

        let mut positions = erasures;
        positions.extend(unknown_positions);
        positions.sort_unstable();
        if positions.is_empty() {
            return Err(ReedSolomonError::UnlocatedErrorsPresent);
        }

        let magnitudes = self.solve_error_magnitudes(
            &positions,
            codeword.len(),
            &syndromes_before[..positions.len()],
        )?;
        self.correct_and_verify(codeword, syndromes_before, &positions, &magnitudes)
    }

    /// Decode under a caller-selected correction envelope.
    ///
    /// The algebraic decoder runs first, but recovered data is released only if
    /// the observed correction counts also satisfy `policy`.
    pub fn decode_with_policy(
        &self,
        codeword: &[u8],
        erasure_positions: &[usize],
        policy: ReedSolomonDecodePolicy,
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        policy.validate(self.config)?;
        if erasure_positions.len() > policy.max_known_erasures {
            return Err(ReedSolomonError::PolicyErasureBudgetExceeded {
                declared: erasure_positions.len(),
                maximum: policy.max_known_erasures,
            });
        }

        let report = self.decode_with_erasures(codeword, erasure_positions)?;
        let corrected = report.corrected_unknown_errors(erasure_positions);
        if corrected > policy.max_unknown_errors {
            return Err(ReedSolomonError::PolicyUnknownErrorBudgetExceeded {
                corrected,
                maximum: policy.max_unknown_errors,
            });
        }
        Ok(report)
    }

    /// Correct up to `floor(parity_symbols / 2)` unknown symbol errors.
    ///
    /// Decoding uses Berlekamp-Massey to infer the error-locator polynomial,
    /// Chien search to recover symbol positions, and an exact Vandermonde solve
    /// for magnitudes. The method returns data only after all configured
    /// syndromes have been recomputed and verified as zero.
    pub fn decode(&self, codeword: &[u8]) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        self.validate_codeword_len(codeword)?;

        let syndromes_before = self.syndromes(codeword)?;
        if syndromes_before.iter().all(|&syndrome| syndrome == 0) {
            return Ok(self.clean_report(codeword, syndromes_before));
        }

        let locator = berlekamp_massey(&syndromes_before)?;
        let locator_degree = locator.len().saturating_sub(1);
        let correction_capacity = self.config.parity_symbols / 2;
        if locator_degree == 0 || locator_degree > correction_capacity {
            return Err(ReedSolomonError::TooManyErrors {
                locator_degree,
                correction_capacity,
            });
        }

        let positions = self.find_error_positions(&locator, codeword.len());
        if positions.len() != locator_degree {
            return Err(ReedSolomonError::LocatorFailure {
                expected: locator_degree,
                found: positions.len(),
            });
        }

        let magnitudes = self.solve_error_magnitudes(
            &positions,
            codeword.len(),
            &syndromes_before[..locator_degree],
        )?;
        self.correct_and_verify(codeword, syndromes_before, &positions, &magnitudes)
    }

    fn validate_codeword_len(&self, codeword: &[u8]) -> Result<(), ReedSolomonError> {
        if codeword.len() > MAX_CODEWORD_LEN {
            return Err(ReedSolomonError::CodewordTooLong {
                codeword_len: codeword.len(),
            });
        }
        if codeword.len() < self.config.parity_symbols {
            return Err(ReedSolomonError::CodewordTooShort {
                codeword_len: codeword.len(),
                parity_symbols: self.config.parity_symbols,
            });
        }
        Ok(())
    }

    fn validate_erasure_positions(
        &self,
        codeword_len: usize,
        erasure_positions: &[usize],
    ) -> Result<Vec<usize>, ReedSolomonError> {
        if erasure_positions.len() > self.config.parity_symbols {
            return Err(ReedSolomonError::TooManyErasures {
                erasures: erasure_positions.len(),
                parity_symbols: self.config.parity_symbols,
            });
        }

        let mut positions = erasure_positions.to_vec();
        positions.sort_unstable();
        for (index, &position) in positions.iter().enumerate() {
            if position >= codeword_len {
                return Err(ReedSolomonError::InvalidErasurePosition {
                    position,
                    codeword_len,
                });
            }
            if index > 0 && positions[index - 1] == position {
                return Err(ReedSolomonError::DuplicateErasurePosition { position });
            }
        }
        Ok(positions)
    }

    fn clean_report(&self, codeword: &[u8], syndromes_before: Vec<u8>) -> ReedSolomonDecodeReport {
        let message_len = codeword.len() - self.config.parity_symbols;
        ReedSolomonDecodeReport {
            message: codeword[..message_len].to_vec(),
            corrected_codeword: codeword.to_vec(),
            corrections: Vec::new(),
            syndromes_before,
        }
    }

    fn remove_erasure_factors(
        &self,
        syndromes: &[u8],
        erasure_positions: &[usize],
        codeword_len: usize,
    ) -> Vec<u8> {
        let mut modified = syndromes.to_vec();
        for &position in erasure_positions {
            let exponent = (codeword_len - 1 - position) % 255;
            let location = pow(self.config.primitive_element, exponent as u32);
            for index in 0..modified.len().saturating_sub(1) {
                modified[index] = add(modified[index + 1], mul(modified[index], location));
            }
            modified.pop();
        }
        modified
    }

    fn correct_and_verify(
        &self,
        codeword: &[u8],
        syndromes_before: Vec<u8>,
        positions: &[usize],
        magnitudes: &[u8],
    ) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
        let mut corrected_codeword = codeword.to_vec();
        let mut corrections = Vec::with_capacity(positions.len());
        for (&position, &magnitude) in positions.iter().zip(magnitudes) {
            corrected_codeword[position] = add(corrected_codeword[position], magnitude);
            if magnitude != 0 {
                corrections.push(SymbolCorrection {
                    position,
                    magnitude,
                });
            }
        }

        if !self.is_valid(&corrected_codeword)? {
            return Err(ReedSolomonError::CorrectionVerificationFailed);
        }

        let message_len = corrected_codeword.len() - self.config.parity_symbols;
        Ok(ReedSolomonDecodeReport {
            message: corrected_codeword[..message_len].to_vec(),
            corrected_codeword,
            corrections,
            syndromes_before,
        })
    }

    fn find_error_positions(&self, locator: &[u8], codeword_len: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(locator.len().saturating_sub(1));
        for position in 0..codeword_len {
            let exponent = (codeword_len - 1 - position) % 255;
            let inverse_exponent = (255 - exponent) % 255;
            let inverse_location = pow(self.config.primitive_element, inverse_exponent as u32);
            if poly_eval_ascending(locator, inverse_location) == 0 {
                positions.push(position);
            }
        }
        positions
    }

    fn solve_error_magnitudes(
        &self,
        positions: &[usize],
        codeword_len: usize,
        leading_syndromes: &[u8],
    ) -> Result<Vec<u8>, ReedSolomonError> {
        let locations = positions
            .iter()
            .map(|&position| {
                let exponent = (codeword_len - 1 - position) % 255;
                pow(self.config.primitive_element, exponent as u32)
            })
            .collect::<Vec<_>>();

        let size = positions.len();
        let mut matrix = vec![vec![0; size]; size];
        for row in 0..size {
            for column in 0..size {
                matrix[row][column] = pow(locations[column], row as u32);
            }
        }

        // Solving S_i = sum(y_j * X_j^i) yields
        // y_j = magnitude_j * X_j^first_root.
        let scaled_magnitudes = solve_linear_system(matrix, leading_syndromes.to_vec())?;
        let mut magnitudes = Vec::with_capacity(size);
        for (scaled, location) in scaled_magnitudes.into_iter().zip(locations) {
            let root_scale = pow(location, u32::from(self.config.first_root));
            magnitudes.push(gf_div(scaled, root_scale)?);
        }
        Ok(magnitudes)
    }

    fn root(&self, offset: usize) -> u8 {
        let exponent = (usize::from(self.config.first_root) + offset) % 255;
        pow(self.config.primitive_element, exponent as u32)
    }
}

/// Checked generator construction using the crate's AES-field convention.
pub fn generator_poly_checked(nsym: usize) -> Result<Vec<u8>, ReedSolomonError> {
    Ok(ReedSolomon::new(ReedSolomonConfig::aes(nsym))?
        .generator
        .clone())
}

/// Checked systematic encoding using the crate's AES-field convention.
pub fn encode_checked(message: &[u8], nsym: usize) -> Result<Vec<u8>, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?.encode(message)
}

/// Checked syndrome calculation using the crate's AES-field convention.
pub fn syndromes_checked(codeword: &[u8], nsym: usize) -> Result<Vec<u8>, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?.syndromes(codeword)
}

/// Checked codeword validation using the crate's AES-field convention.
pub fn is_valid_checked(codeword: &[u8], nsym: usize) -> Result<bool, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?.is_valid(codeword)
}

/// Checked bounded-distance decoding using the crate's AES-field convention.
pub fn decode_checked(
    codeword: &[u8],
    nsym: usize,
) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?.decode(codeword)
}

/// Checked known-erasure recovery using the crate's AES-field convention.
pub fn decode_erasures_checked(
    codeword: &[u8],
    nsym: usize,
    erasure_positions: &[usize],
) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?.decode_erasures(codeword, erasure_positions)
}

/// Checked mixed unknown-error and known-erasure recovery using the default field convention.
pub fn decode_with_erasures_checked(
    codeword: &[u8],
    nsym: usize,
    erasure_positions: &[usize],
) -> Result<ReedSolomonDecodeReport, ReedSolomonError> {
    ReedSolomon::new(ReedSolomonConfig::aes(nsym))?
        .decode_with_erasures(codeword, erasure_positions)
}

/// Compatibility wrapper for generator construction.
///
/// # Panics
/// Panics when `nsym` is outside `1..=254`. New code should use
/// [`generator_poly_checked`].
#[must_use]
pub fn generator_poly(nsym: usize) -> Vec<u8> {
    generator_poly_checked(nsym).expect("invalid Reed-Solomon parity count")
}

/// Compatibility wrapper for systematic encoding.
///
/// # Panics
/// Panics for an invalid parity count or a codeword longer than 255 symbols.
/// New code should use [`encode_checked`].
#[must_use]
pub fn encode(message: &[u8], nsym: usize) -> Vec<u8> {
    encode_checked(message, nsym).expect("invalid Reed-Solomon encoding parameters")
}

/// Compatibility wrapper for syndrome calculation.
///
/// # Panics
/// Panics for an invalid parity count or a codeword longer than 255 symbols.
/// New code should use [`syndromes_checked`].
#[must_use]
pub fn syndromes(codeword: &[u8], nsym: usize) -> Vec<u8> {
    syndromes_checked(codeword, nsym).expect("invalid Reed-Solomon syndrome parameters")
}

/// Compatibility wrapper for codeword validation.
///
/// Invalid configurations return `false` instead of accepting a word.
#[must_use]
pub fn is_valid(codeword: &[u8], nsym: usize) -> bool {
    is_valid_checked(codeword, nsym).unwrap_or(false)
}

fn berlekamp_massey(syndromes: &[u8]) -> Result<Vec<u8>, ReedSolomonError> {
    // Coefficients are low degree first: Λ(z) = 1 + λ₁z + ... + λₗzˡ.
    let mut locator = vec![1u8];
    let mut previous = vec![1u8];
    let mut locator_degree = 0usize;
    let mut shift = 1usize;
    let mut previous_discrepancy = 1u8;

    for step in 0..syndromes.len() {
        let mut discrepancy = syndromes[step];
        for coefficient in 1..=locator_degree {
            if coefficient < locator.len() {
                discrepancy = add(
                    discrepancy,
                    mul(locator[coefficient], syndromes[step - coefficient]),
                );
            }
        }

        if discrepancy == 0 {
            shift += 1;
            continue;
        }

        let snapshot = locator.clone();
        let scale = gf_div(discrepancy, previous_discrepancy)?;
        let required_len = previous.len() + shift;
        if locator.len() < required_len {
            locator.resize(required_len, 0);
        }
        for (index, &coefficient) in previous.iter().enumerate() {
            let target = index + shift;
            locator[target] = add(locator[target], mul(scale, coefficient));
        }

        if 2 * locator_degree <= step {
            locator_degree = step + 1 - locator_degree;
            previous = snapshot;
            previous_discrepancy = discrepancy;
            shift = 1;
        } else {
            shift += 1;
        }
    }

    while locator.len() > 1 && locator.last() == Some(&0) {
        locator.pop();
    }
    Ok(locator)
}

fn solve_linear_system(
    matrix: Vec<Vec<u8>>,
    right_hand_side: Vec<u8>,
) -> Result<Vec<u8>, ReedSolomonError> {
    let size = right_hand_side.len();
    if matrix.len() != size || matrix.iter().any(|row| row.len() != size) {
        return Err(ReedSolomonError::SingularMagnitudeSystem);
    }

    let mut augmented = matrix
        .into_iter()
        .zip(right_hand_side)
        .map(|(mut row, value)| {
            row.push(value);
            row
        })
        .collect::<Vec<_>>();

    for column in 0..size {
        let pivot = (column..size)
            .find(|&row| augmented[row][column] != 0)
            .ok_or(ReedSolomonError::SingularMagnitudeSystem)?;
        augmented.swap(column, pivot);

        let inverse = gf_inverse(augmented[column][column])?;
        for entry in &mut augmented[column] {
            *entry = mul(*entry, inverse);
        }

        for row in 0..size {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor == 0 {
                continue;
            }
            for entry in column..=size {
                augmented[row][entry] =
                    add(augmented[row][entry], mul(factor, augmented[column][entry]));
            }
        }
    }

    Ok((0..size).map(|row| augmented[row][size]).collect())
}

fn gf_inverse(value: u8) -> Result<u8, ReedSolomonError> {
    if value == 0 {
        return Err(ReedSolomonError::SingularMagnitudeSystem);
    }
    Ok(pow(value, 254))
}

fn gf_div(numerator: u8, denominator: u8) -> Result<u8, ReedSolomonError> {
    if numerator == 0 {
        return Ok(0);
    }
    Ok(mul(numerator, gf_inverse(denominator)?))
}

/// Evaluate a low-degree-first polynomial over GF(2⁸).
fn poly_eval_ascending(poly: &[u8], x: u8) -> u8 {
    poly.iter().rev().fold(0, |accumulator, &coefficient| {
        add(mul(accumulator, x), coefficient)
    })
}

fn has_order_255(element: u8) -> bool {
    // 255 = 3 * 5 * 17. An element has order 255 iff a^255 = 1 and it
    // survives each maximal proper-factor test.
    pow(element, 255) == 1
        && pow(element, 255 / 3) != 1
        && pow(element, 255 / 5) != 1
        && pow(element, 255 / 17) != 1
}

fn build_generator(config: ReedSolomonConfig) -> Vec<u8> {
    let mut generator = vec![1];
    for offset in 0..config.parity_symbols {
        let exponent = (usize::from(config.first_root) + offset) % 255;
        let root = pow(config.primitive_element, exponent as u32);
        generator = poly_mul(&generator, &[1, root]);
    }
    generator
}

/// Polynomial product over GF(2⁸), most-significant coefficient first.
fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = vec![0; a.len() + b.len() - 1];
    for (i, &left) in a.iter().enumerate() {
        for (j, &right) in b.iter().enumerate() {
            result[i + j] = add(result[i + j], mul(left, right));
        }
    }
    result
}

/// Evaluate a polynomial with Horner's method over GF(2⁸).
fn poly_eval(poly: &[u8], x: u8) -> u8 {
    poly.iter().fold(0, |accumulator, &coefficient| {
        add(mul(accumulator, x), coefficient)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_generator_is_primitive_but_x_is_not() {
        assert!(has_order_255(AES_PRIMITIVE_ELEMENT));
        assert!(!has_order_255(0x02));
        assert_eq!(pow(0x02, 51), 1);
    }

    #[test]
    fn encoded_word_is_valid_and_systematic() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let message = [0x12, 0x34, 0x56];
        let codeword = codec.encode(&message).unwrap();
        assert_eq!(codeword.len(), message.len() + 4);
        assert_eq!(&codeword[..message.len()], &message);
        assert!(codec.is_valid(&codeword).unwrap());
    }

    #[test]
    fn parameters_and_fixed_frames_bind_k_n_and_distance() {
        let frame = ReedSolomonFrame::new(ReedSolomonConfig::aes(8), 23).unwrap();
        let parameters = frame.parameters();
        assert_eq!(parameters.message_symbols, 23);
        assert_eq!(parameters.parity_symbols, 8);
        assert_eq!(parameters.codeword_symbols, 31);
        assert_eq!(parameters.minimum_distance, 9);
        assert!(parameters.supports_errata(3, 2));
        assert!(!parameters.supports_errata(4, 1));

        let message = vec![0xA5; 23];
        let clean = frame.encode(&message).unwrap();
        let mut corrupted = clean.clone();
        corrupted[2] ^= 0x11;
        corrupted[17] ^= 0x88;
        assert_eq!(frame.decode(&corrupted).unwrap().message, message);

        assert_eq!(
            frame.encode(&message[..22]),
            Err(ReedSolomonError::MessageLengthMismatch {
                expected: 23,
                actual: 22,
            })
        );
        assert_eq!(
            frame.decode(&clean[..30]),
            Err(ReedSolomonError::CodewordLengthMismatch {
                expected: 31,
                actual: 30,
            })
        );
    }

    #[test]
    fn shortened_encoding_matches_zero_prefixed_parent_code() {
        let shortened =
            ReedSolomonShortenedFrame::new(ReedSolomonConfig::aes(32), 223, 19).unwrap();
        let message = (0u8..19).collect::<Vec<_>>();
        let short_codeword = shortened.encode(&message).unwrap();
        let parent_message = shortened.expand_message(&message).unwrap();
        let parent_codeword = shortened.frame().codec().encode(&parent_message).unwrap();

        assert_eq!(shortened.shortening_symbols(), 204);
        assert!(parent_codeword[..204].iter().all(|&symbol| symbol == 0));
        assert_eq!(&parent_codeword[204..], short_codeword);
        assert_eq!(
            shortened.expand_codeword(&short_codeword).unwrap(),
            parent_codeword
        );
        assert_eq!(
            shortened
                .contract_parent_codeword(&parent_codeword)
                .unwrap(),
            short_codeword
        );
        assert_eq!(shortened.parameters().minimum_distance, 33);
        assert_eq!(shortened.parent_parameters().codeword_symbols, 255);
    }

    #[test]
    fn shortened_frame_rejects_invalid_parent_contracts() {
        assert_eq!(
            ReedSolomonShortenedFrame::new(ReedSolomonConfig::aes(8), 12, 13),
            Err(ReedSolomonError::InvalidShortening {
                parent_message_symbols: 12,
                transmitted_message_symbols: 13,
            })
        );

        let shortened = ReedSolomonShortenedFrame::new(ReedSolomonConfig::aes(8), 32, 12).unwrap();
        let mut parent = shortened
            .expand_codeword(&shortened.encode(&[0xA5; 12]).unwrap())
            .unwrap();
        parent[7] = 0x44;
        assert_eq!(
            shortened.contract_parent_codeword(&parent),
            Err(ReedSolomonError::NonZeroShorteningPrefix {
                position: 7,
                value: 0x44,
            })
        );
    }

    #[test]
    fn shortened_frame_decodes_in_transmitted_coordinates() {
        let shortened =
            ReedSolomonShortenedFrame::new(ReedSolomonConfig::aes(12), 200, 21).unwrap();
        let message = b"shortened coordinates";
        let clean = shortened.encode(message).unwrap();
        let erasures = [0usize, clean.len() - 1];
        let mut corrupted = clean.clone();
        corrupted[0] = 0;
        let last = corrupted.len() - 1;
        corrupted[last] = 0xFF;
        corrupted[7] ^= 0x5A;

        let report = shortened
            .decode_with_policy(&corrupted, &erasures, ReedSolomonDecodePolicy::new(1, 2))
            .unwrap();
        assert_eq!(report.message, message);
        assert_eq!(report.corrected_codeword, clean);
    }

    #[test]
    fn streaming_parity_matches_systematic_encoder_at_every_split() {
        for first_root in [0u8, 1, 17] {
            let codec = ReedSolomon::new(ReedSolomonConfig {
                parity_symbols: 12,
                primitive_element: AES_PRIMITIVE_ELEMENT,
                first_root,
            })
            .unwrap();
            let message = b"streaming parity must be independent of chunk boundaries";
            let expected = codec.encode(message).unwrap();
            let expected_parity = &expected[message.len()..];

            assert_eq!(codec.encode_parity(message).unwrap(), expected_parity);
            for split in 0..=message.len() {
                let mut state = codec.parity_state();
                state.update(&message[..split]).unwrap();
                state.update(&message[split..]).unwrap();
                assert_eq!(state.message_symbols(), message.len());
                assert_eq!(state.parity(), expected_parity);
            }

            let mut symbolwise = codec.parity_state();
            for &symbol in message {
                symbolwise.update_symbol(symbol).unwrap();
            }
            assert_eq!(symbolwise.finalize(), expected_parity);
        }
    }

    #[test]
    fn streaming_state_resets_and_rejects_overflow_without_mutation() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        let mut state = codec.parity_state();
        state.update(b"first message").unwrap();
        assert_ne!(state.parity(), &[0; 8]);
        state.reset();
        assert_eq!(state.message_symbols(), 0);
        assert_eq!(state.parity(), &[0; 8]);

        let maximum = vec![0x5A; MAX_CODEWORD_LEN - codec.config().parity_symbols];
        state.update(&maximum).unwrap();
        let before = state.clone();
        assert_eq!(
            state.update_symbol(0xA5),
            Err(ReedSolomonError::MessageTooLong {
                message_len: maximum.len() + 1,
                parity_symbols: 8,
            })
        );
        assert_eq!(state, before);
    }

    #[test]
    fn parity_buffers_are_exact_and_match_golden_encoding() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let message = [0x12, 0x34, 0x56];
        let mut parity = [0u8; 4];
        codec.encode_parity_into(&message, &mut parity).unwrap();
        assert_eq!(parity, [0x33, 0xE0, 0xE4, 0x47]);
        assert_eq!(
            codec.encode_parity_into(&message, &mut [0; 3]),
            Err(ReedSolomonError::ParityLengthMismatch {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn caller_owned_encoding_buffer_matches_allocating_api() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        let message = (0u8..48).collect::<Vec<_>>();
        let expected = codec.encode(&message).unwrap();
        let mut output = vec![0xAA; codec.encoded_len(message.len()).unwrap()];
        codec.encode_into(&message, &mut output).unwrap();
        assert_eq!(output, expected);

        let mut too_short = vec![0; output.len() - 1];
        assert_eq!(
            codec.encode_into(&message, &mut too_short),
            Err(ReedSolomonError::OutputLengthMismatch {
                expected: output.len(),
                actual: output.len() - 1,
            })
        );
    }

    #[test]
    fn detects_single_symbol_corruption() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let mut codeword = codec.encode(&[0xAA, 0xBB, 0xCC, 0xDD]).unwrap();
        codeword[2] ^= 0x01;
        assert!(!codec.is_valid(&codeword).unwrap());
    }

    #[test]
    fn detects_the_old_order_51_alias_pattern() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let mut codeword = codec.encode(&[0x5A; 60]).unwrap();
        codeword[0] ^= 0xA7;
        codeword[51] ^= 0xA7;
        assert!(!codec.is_valid(&codeword).unwrap());
    }

    #[test]
    fn generator_has_expected_degree_and_roots() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        assert_eq!(codec.generator().len(), 9);
        for offset in 0..8 {
            assert_eq!(poly_eval(codec.generator(), codec.root(offset)), 0);
        }
    }

    #[test]
    fn rejects_invalid_field_and_length_parameters() {
        assert_eq!(
            ReedSolomon::new(ReedSolomonConfig::aes(0)),
            Err(ReedSolomonError::ZeroParitySymbols)
        );
        assert_eq!(
            ReedSolomon::new(ReedSolomonConfig {
                parity_symbols: 4,
                primitive_element: 0x02,
                first_root: 0,
            }),
            Err(ReedSolomonError::NonPrimitiveElement { element: 0x02 })
        );
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(32)).unwrap();
        assert!(matches!(
            codec.encode(&[0; 224]),
            Err(ReedSolomonError::MessageTooLong { .. })
        ));
    }

    #[test]
    fn compatibility_wrappers_use_the_validated_default() {
        let message = [1, 2, 3, 4];
        let codeword = encode(&message, 4);
        assert!(is_valid(&codeword, 4));
        assert_eq!(generator_poly(4).len(), 5);
    }

    #[test]
    fn decoder_returns_clean_words_without_corrections() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        let message = [1, 3, 3, 7, 9];
        let codeword = codec.encode(&message).unwrap();
        let report = codec.decode(&codeword).unwrap();
        assert_eq!(report.message, message);
        assert_eq!(report.corrected_codeword, codeword);
        assert!(report.corrections.is_empty());
        assert!(report.syndromes_before.iter().all(|&value| value == 0));
    }

    #[test]
    fn decoder_corrects_every_single_position_and_magnitude() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let message = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90];
        let codeword = codec.encode(&message).unwrap();
        for position in 0..codeword.len() {
            for magnitude in 1u16..=255 {
                let mut corrupted = codeword.clone();
                corrupted[position] ^= magnitude as u8;
                let report = codec.decode(&corrupted).unwrap();
                assert_eq!(report.message, message);
                assert_eq!(report.corrected_codeword, codeword);
                assert_eq!(
                    report.corrections,
                    [SymbolCorrection {
                        position,
                        magnitude: magnitude as u8,
                    }]
                );
            }
        }
    }

    #[test]
    fn decoder_corrects_seeded_patterns_through_capacity() {
        let mut rng = Lcg::new(0xC0D1_7EED_5EED_1234);
        for parity_symbols in [2usize, 4, 6, 8, 10, 16] {
            let codec = ReedSolomon::new(ReedSolomonConfig::aes(parity_symbols)).unwrap();
            for message_len in [1usize, 3, 17, 63, 255 - parity_symbols] {
                for error_count in 0..=parity_symbols / 2 {
                    for _case in 0..12 {
                        let message = (0..message_len).map(|_| rng.next_u8()).collect::<Vec<_>>();
                        let codeword = codec.encode(&message).unwrap();
                        let mut corrupted = codeword.clone();
                        let positions = unique_positions(&mut rng, codeword.len(), error_count);
                        for &position in &positions {
                            let mut magnitude = rng.next_u8();
                            if magnitude == 0 {
                                magnitude = 1;
                            }
                            corrupted[position] ^= magnitude;
                        }

                        let report = codec.decode(&corrupted).unwrap();
                        assert_eq!(report.message, message);
                        assert_eq!(report.corrected_codeword, codeword);
                        assert_eq!(report.corrections.len(), error_count);
                    }
                }
            }
        }
    }

    #[test]
    fn erasure_decoder_recovers_every_single_position_and_magnitude() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let message = [0x10, 0x20, 0x30, 0x40, 0x50];
        let codeword = codec.encode(&message).unwrap();

        for position in 0..codeword.len() {
            for magnitude in 1u16..=255 {
                let mut erased = codeword.clone();
                erased[position] ^= magnitude as u8;
                let report = codec.decode_erasures(&erased, &[position]).unwrap();
                assert_eq!(report.message, message);
                assert_eq!(report.corrected_codeword, codeword);
                assert_eq!(
                    report.corrections,
                    [SymbolCorrection {
                        position,
                        magnitude: magnitude as u8,
                    }]
                );
            }
        }
    }

    #[test]
    fn erasure_decoder_recovers_through_full_parity_capacity() {
        let mut rng = Lcg::new(0xE2A5_AE55_5EED_u64);
        for parity_symbols in [2usize, 4, 8, 12] {
            let codec = ReedSolomon::new(ReedSolomonConfig::aes(parity_symbols)).unwrap();
            for erasure_count in 0..=parity_symbols {
                for _case in 0..10 {
                    let message = (0..31).map(|_| rng.next_u8()).collect::<Vec<_>>();
                    let codeword = codec.encode(&message).unwrap();
                    let positions = unique_positions(&mut rng, codeword.len(), erasure_count);
                    let mut erased = codeword.clone();
                    for &position in &positions {
                        erased[position] = rng.next_u8();
                    }

                    let report = codec.decode_erasures(&erased, &positions).unwrap();
                    assert_eq!(report.message, message);
                    assert_eq!(report.corrected_codeword, codeword);
                }
            }
        }
    }

    #[test]
    fn erasure_decoder_rejects_invalid_location_sets() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(4)).unwrap();
        let codeword = codec.encode(&[1, 2, 3]).unwrap();
        assert_eq!(
            codec.decode_erasures(&codeword, &[1, 1]),
            Err(ReedSolomonError::DuplicateErasurePosition { position: 1 })
        );
        assert_eq!(
            codec.decode_erasures(&codeword, &[codeword.len()]),
            Err(ReedSolomonError::InvalidErasurePosition {
                position: codeword.len(),
                codeword_len: codeword.len(),
            })
        );
        assert_eq!(
            codec.decode_erasures(&codeword, &[0, 1, 2, 3, 4]),
            Err(ReedSolomonError::TooManyErasures {
                erasures: 5,
                parity_symbols: 4,
            })
        );
    }

    #[test]
    fn erasure_decoder_fails_closed_on_unlocated_errors() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(6)).unwrap();
        let mut codeword = codec.encode(&[1, 2, 3, 4, 5]).unwrap();
        codeword[1] ^= 0x55;
        assert_eq!(
            codec.decode_erasures(&codeword, &[]),
            Err(ReedSolomonError::UnlocatedErrorsPresent)
        );
        assert_eq!(
            codec.decode_erasures(&codeword, &[0]),
            Err(ReedSolomonError::CorrectionVerificationFailed)
        );
    }

    #[test]
    fn mixed_decoder_recovers_every_capacity_partition() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(10)).unwrap();
        let message = b"mixed errors and erasures";
        let clean = codec.encode(message).unwrap();
        let mut rng = Lcg::new(0xE22A_5EED);

        for erasures in 0..=10 {
            for errors in 0..=(10 - erasures) / 2 {
                for _ in 0..24 {
                    let positions = unique_positions(&mut rng, clean.len(), erasures + errors);
                    let erasure_positions = &positions[..erasures];
                    let mut corrupted = clean.clone();
                    for &position in erasure_positions {
                        corrupted[position] = rng.next_u8();
                    }
                    for &position in &positions[erasures..] {
                        let mut magnitude = rng.next_u8();
                        if magnitude == 0 {
                            magnitude = 1;
                        }
                        corrupted[position] ^= magnitude;
                    }

                    let report = codec
                        .decode_with_erasures(&corrupted, erasure_positions)
                        .unwrap();
                    assert_eq!(report.message, message);
                    assert_eq!(report.corrected_codeword, clean);
                }
            }
        }
    }

    #[test]
    fn mixed_decoder_supports_nonzero_first_root() {
        let codec = ReedSolomon::new(ReedSolomonConfig {
            parity_symbols: 8,
            primitive_element: AES_PRIMITIVE_ELEMENT,
            first_root: 17,
        })
        .unwrap();
        let message = b"root-aware mixed decoding";
        let clean = codec.encode(message).unwrap();
        let mut corrupted = clean.clone();
        let erasures = [1usize, 9];
        corrupted[1] = 0x00;
        corrupted[9] = 0xFF;
        corrupted[4] ^= 0x53;
        corrupted[17] ^= 0xA6;

        let report = codec.decode_with_erasures(&corrupted, &erasures).unwrap();
        assert_eq!(report.message, message);
        assert_eq!(report.corrected_codeword, clean);
    }

    #[test]
    fn mixed_decoder_fails_closed_beyond_capacity() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(6)).unwrap();
        let message = b"capacity boundary";
        let clean = codec.encode(message).unwrap();
        let erasures = [0usize, 1, 2];
        let mut observed_failure = false;

        for seed in 0..64u64 {
            let mut corrupted = clean.clone();
            corrupted[0] = seed as u8;
            corrupted[1] = (seed as u8).wrapping_mul(3);
            corrupted[2] = (seed as u8).wrapping_add(17);
            corrupted[5] ^= 0x55;
            corrupted[9] ^= 0xA7;
            let result = codec.decode_with_erasures(&corrupted, &erasures);
            if result.is_err() {
                observed_failure = true;
                break;
            }
        }

        assert!(
            observed_failure,
            "beyond-capacity corruption must not be universally accepted"
        );
    }

    #[test]
    fn decode_policy_must_fit_the_algebraic_errata_bound() {
        let config = ReedSolomonConfig::aes(8);
        assert!(ReedSolomonDecodePolicy::new(2, 4).validate(config).is_ok());
        assert_eq!(
            ReedSolomonDecodePolicy::new(3, 3).validate(config),
            Err(ReedSolomonError::InvalidDecodePolicy {
                max_unknown_errors: 3,
                max_known_erasures: 3,
                parity_symbols: 8,
            })
        );
    }

    #[test]
    fn decode_policy_rejects_corrections_beyond_caller_budget() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        let clean = codec.encode(b"policy-bound decoding").unwrap();
        let mut corrupted = clean.clone();
        corrupted[2] ^= 0x51;
        corrupted[11] ^= 0xA6;

        assert_eq!(
            codec.decode_with_policy(&corrupted, &[], ReedSolomonDecodePolicy::new(1, 0),),
            Err(ReedSolomonError::PolicyUnknownErrorBudgetExceeded {
                corrected: 2,
                maximum: 1,
            })
        );

        let accepted = codec
            .decode_with_policy(&corrupted, &[], ReedSolomonDecodePolicy::new(2, 0))
            .unwrap();
        assert_eq!(accepted.corrected_codeword, clean);
        assert_eq!(accepted.corrected_unknown_errors(&[]), 2);
        assert_eq!(accepted.corrected_errata_weight(&[]), 4);
    }

    #[test]
    fn decode_policy_accounts_for_declared_erasures_separately() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        let clean = codec.encode(b"mixed policy").unwrap();
        let erasures = [1usize, 5];
        let mut corrupted = clean.clone();
        corrupted[1] = 0;
        corrupted[5] = 0xFF;
        corrupted[9] ^= 0x77;

        assert_eq!(
            codec.decode_with_policy(&corrupted, &erasures, ReedSolomonDecodePolicy::new(2, 1),),
            Err(ReedSolomonError::PolicyErasureBudgetExceeded {
                declared: 2,
                maximum: 1,
            })
        );

        let report = codec
            .decode_with_policy(&corrupted, &erasures, ReedSolomonDecodePolicy::new(1, 2))
            .unwrap();
        assert_eq!(report.corrected_unknown_errors(&erasures), 1);
        assert_eq!(report.corrected_erasures(&erasures), 2);
        assert_eq!(report.corrected_errata_weight(&erasures), 4);
    }

    #[test]
    fn decoder_supports_nonzero_first_root() {
        let codec = ReedSolomon::new(ReedSolomonConfig {
            parity_symbols: 8,
            primitive_element: AES_PRIMITIVE_ELEMENT,
            first_root: 17,
        })
        .unwrap();
        let message = (0u8..40).collect::<Vec<_>>();
        let codeword = codec.encode(&message).unwrap();
        let mut corrupted = codeword.clone();
        corrupted[0] ^= 0xA1;
        corrupted[19] ^= 0xB2;
        corrupted[47] ^= 0xC3;
        let report = codec.decode(&corrupted).unwrap();
        assert_eq!(report.message, message);
        assert_eq!(report.corrected_codeword, codeword);
    }

    #[test]
    fn decoder_rejects_words_shorter_than_the_parity_tail() {
        let codec = ReedSolomon::new(ReedSolomonConfig::aes(8)).unwrap();
        assert_eq!(
            codec.is_valid(&[1, 2, 3]),
            Err(ReedSolomonError::CodewordTooShort {
                codeword_len: 3,
                parity_symbols: 8,
            })
        );
        assert_eq!(
            codec.decode(&[1, 2, 3]),
            Err(ReedSolomonError::CodewordTooShort {
                codeword_len: 3,
                parity_symbols: 8,
            })
        );
    }

    struct Lcg(u64);

    impl Lcg {
        const fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn next_u8(&mut self) -> u8 {
            (self.next_u64() >> 56) as u8
        }

        fn index(&mut self, upper: usize) -> usize {
            (self.next_u64() as usize) % upper
        }
    }

    fn unique_positions(rng: &mut Lcg, len: usize, count: usize) -> Vec<usize> {
        let mut positions = Vec::with_capacity(count);
        while positions.len() < count {
            let candidate = rng.index(len);
            if !positions.contains(&candidate) {
                positions.push(candidate);
            }
        }
        positions
    }
}
