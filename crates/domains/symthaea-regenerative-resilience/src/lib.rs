// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, authority-free primitives for REGEN compound-shock modeling.
//!
//! This crate begins the Symthaea-side REGEN-042S implementation with only the
//! canonical identity, quantity, ratio, and time types needed before any shock,
//! dependency, recovery, or service-transition algorithm is allowed to exist.
//!
//! The core deliberately performs no I/O, networking, governance, authorization,
//! physical control, or live Mycelix lookup. External/adopted identities are
//! carried opaquely and validated only for canonical representation.

#![forbid(unsafe_code)]

use std::fmt;
use std::num::NonZeroU64;

/// Maximum UTF-8 byte length accepted by the v1 canonical identifier grammar.
pub const MAX_ID_BYTES: usize = 128;
/// Maximum decimal scale supported by [`CanonicalQuantity`].
pub const MAX_DECIMAL_SCALE: u8 = 18;

/// Failure to construct a canonical cross-system identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentifierError {
    Empty,
    TooLong { actual: usize, maximum: usize },
    NonAscii { index: usize },
    InvalidByte { index: usize, byte: u8 },
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "identifier is empty"),
            Self::TooLong { actual, maximum } => {
                write!(f, "identifier length {actual} exceeds maximum {maximum}")
            }
            Self::NonAscii { index } => write!(f, "identifier contains non-ASCII byte at {index}"),
            Self::InvalidByte { index, byte } => {
                write!(f, "identifier contains invalid byte 0x{byte:02x} at {index}")
            }
        }
    }
}

impl std::error::Error for IdentifierError {}

/// Opaque canonical identifier used to preserve an upstream identity exactly.
///
/// The grammar is deliberately narrow and normalization-free: callers must pass
/// the exact identifier revision they intend to bind. Case folding, Unicode
/// normalization, whitespace trimming, and path normalization are prohibited.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalId(String);

impl CanonicalId {
    pub fn new(value: impl Into<String>) -> Result<Self, IdentifierError> {
        let value = value.into();
        if value.is_empty() {
            return Err(IdentifierError::Empty);
        }
        if value.len() > MAX_ID_BYTES {
            return Err(IdentifierError::TooLong {
                actual: value.len(),
                maximum: MAX_ID_BYTES,
            });
        }
        for (index, byte) in value.bytes().enumerate() {
            if !byte.is_ascii() {
                return Err(IdentifierError::NonAscii { index });
            }
            let valid = byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/');
            if !valid {
                return Err(IdentifierError::InvalidByte { index, byte });
            }
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CanonicalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

macro_rules! canonical_id_newtype {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(CanonicalId);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdentifierError> {
                CanonicalId::new(value).map(Self)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }

            pub fn canonical(&self) -> &CanonicalId {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

canonical_id_newtype!(ServiceId);
canonical_id_newtype!(DependencyId);
canonical_id_newtype!(StockId);
canonical_id_newtype!(FailureDomainId);
canonical_id_newtype!(EventId);
canonical_id_newtype!(RecoveryResourceId);
canonical_id_newtype!(CampaignRevisionId);
canonical_id_newtype!(ProfileRevisionId);
canonical_id_newtype!(EvidenceSnapshotId);
canonical_id_newtype!(ModelRevisionId);
canonical_id_newtype!(UnitId);
canonical_id_newtype!(BasisId);
canonical_id_newtype!(CommodityId);
canonical_id_newtype!(ScopeId);
canonical_id_newtype!(ProvisionPathId);

/// Exact-quantity validation/arithmetic failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantityError {
    ScaleTooLarge { scale: u8, maximum: u8 },
    UnitMismatch { left: UnitId, right: UnitId },
    BasisMismatch { left: BasisId, right: BasisId },
    Overflow,
    NegativeQuantity,
    ZeroRatioDenominator,
}

impl fmt::Display for QuantityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScaleTooLarge { scale, maximum } => {
                write!(f, "decimal scale {scale} exceeds maximum {maximum}")
            }
            Self::UnitMismatch { left, right } => {
                write!(f, "quantity unit mismatch: {left} != {right}")
            }
            Self::BasisMismatch { left, right } => {
                write!(f, "quantity basis mismatch: {left} != {right}")
            }
            Self::Overflow => write!(f, "exact quantity arithmetic overflow"),
            Self::NegativeQuantity => write!(f, "quantity must be non-negative"),
            Self::ZeroRatioDenominator => write!(f, "ratio denominator must be non-zero"),
        }
    }
}

impl std::error::Error for QuantityError {}

/// Exact fixed-decimal quantity with explicit unit and basis identity.
///
/// Value semantics are `mantissa * 10^-scale`. Construction canonicalizes away
/// trailing decimal zeros, so equal values with the same unit/basis have one
/// preferred in-memory decimal representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalQuantity {
    mantissa: i128,
    scale: u8,
    unit: UnitId,
    basis: BasisId,
}

impl CanonicalQuantity {
    pub fn new(
        mantissa: i128,
        scale: u8,
        unit: UnitId,
        basis: BasisId,
    ) -> Result<Self, QuantityError> {
        if scale > MAX_DECIMAL_SCALE {
            return Err(QuantityError::ScaleTooLarge {
                scale,
                maximum: MAX_DECIMAL_SCALE,
            });
        }
        let mut quantity = Self {
            mantissa,
            scale,
            unit,
            basis,
        };
        quantity.normalize();
        Ok(quantity)
    }

    fn normalize(&mut self) {
        if self.mantissa == 0 {
            self.scale = 0;
            return;
        }
        while self.scale > 0 && self.mantissa % 10 == 0 {
            self.mantissa /= 10;
            self.scale -= 1;
        }
    }

    pub fn mantissa(&self) -> i128 {
        self.mantissa
    }

    pub fn scale(&self) -> u8 {
        self.scale
    }

    pub fn unit(&self) -> &UnitId {
        &self.unit
    }

    pub fn basis(&self) -> &BasisId {
        &self.basis
    }

    pub fn is_negative(&self) -> bool {
        self.mantissa < 0
    }

    fn require_compatible(&self, other: &Self) -> Result<(), QuantityError> {
        if self.unit != other.unit {
            return Err(QuantityError::UnitMismatch {
                left: self.unit.clone(),
                right: other.unit.clone(),
            });
        }
        if self.basis != other.basis {
            return Err(QuantityError::BasisMismatch {
                left: self.basis.clone(),
                right: other.basis.clone(),
            });
        }
        Ok(())
    }

    fn mantissa_at_scale(&self, target_scale: u8) -> Result<i128, QuantityError> {
        debug_assert!(target_scale >= self.scale);
        let exponent = u32::from(target_scale - self.scale);
        let multiplier = 10_i128
            .checked_pow(exponent)
            .ok_or(QuantityError::Overflow)?;
        self.mantissa
            .checked_mul(multiplier)
            .ok_or(QuantityError::Overflow)
    }

    pub fn checked_add(&self, other: &Self) -> Result<Self, QuantityError> {
        self.require_compatible(other)?;
        let scale = self.scale.max(other.scale);
        let left = self.mantissa_at_scale(scale)?;
        let right = other.mantissa_at_scale(scale)?;
        let mantissa = left.checked_add(right).ok_or(QuantityError::Overflow)?;
        Self::new(mantissa, scale, self.unit.clone(), self.basis.clone())
    }

    pub fn checked_sub(&self, other: &Self) -> Result<Self, QuantityError> {
        self.require_compatible(other)?;
        let scale = self.scale.max(other.scale);
        let left = self.mantissa_at_scale(scale)?;
        let right = other.mantissa_at_scale(scale)?;
        let mantissa = left.checked_sub(right).ok_or(QuantityError::Overflow)?;
        Self::new(mantissa, scale, self.unit.clone(), self.basis.clone())
    }
}

/// Exact quantity constrained to the non-negative domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonNegativeQuantity(CanonicalQuantity);

impl NonNegativeQuantity {
    pub fn try_new(quantity: CanonicalQuantity) -> Result<Self, QuantityError> {
        if quantity.is_negative() {
            return Err(QuantityError::NegativeQuantity);
        }
        Ok(Self(quantity))
    }

    pub fn new(
        mantissa: i128,
        scale: u8,
        unit: UnitId,
        basis: BasisId,
    ) -> Result<Self, QuantityError> {
        Self::try_new(CanonicalQuantity::new(mantissa, scale, unit, basis)?)
    }

    pub fn quantity(&self) -> &CanonicalQuantity {
        &self.0
    }

    pub fn checked_add(&self, other: &Self) -> Result<Self, QuantityError> {
        Self::try_new(self.0.checked_add(&other.0)?)
    }

    pub fn checked_sub(&self, other: &Self) -> Result<Self, QuantityError> {
        Self::try_new(self.0.checked_sub(&other.0)?)
    }
}

/// Exact rational multiplier used for capacity/demand scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactRatio {
    numerator: u64,
    denominator: NonZeroU64,
}

impl ExactRatio {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, QuantityError> {
        let denominator =
            NonZeroU64::new(denominator).ok_or(QuantityError::ZeroRatioDenominator)?;
        let divisor = gcd(numerator, denominator.get());
        let numerator = numerator / divisor;
        let denominator = NonZeroU64::new(denominator.get() / divisor)
            .expect("a non-zero integer divided by a positive divisor remains non-zero");
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub fn numerator(self) -> u64 {
        self.numerator
    }

    pub fn denominator(self) -> u64 {
        self.denominator.get()
    }

    /// Apply this ratio to a non-negative exact quantity using floor rounding
    /// at the quantity's existing decimal scale.
    pub fn apply_floor(
        self,
        quantity: &NonNegativeQuantity,
    ) -> Result<NonNegativeQuantity, QuantityError> {
        let numerator = i128::from(self.numerator);
        let denominator = i128::from(self.denominator.get());
        let scaled = quantity
            .0
            .mantissa
            .checked_mul(numerator)
            .ok_or(QuantityError::Overflow)?;
        let mantissa = scaled / denominator;
        NonNegativeQuantity::new(
            mantissa,
            quantity.0.scale,
            quantity.0.unit.clone(),
            quantity.0.basis.clone(),
        )
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    if left == 0 {
        return right.max(1);
    }
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Canonical campaign time tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tick(pub u64);

/// Non-empty half-open interval `[start, end_exclusive)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeInterval {
    start: Tick,
    end_exclusive: Tick,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeError {
    EmptyOrReversedInterval { start: Tick, end_exclusive: Tick },
}

impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOrReversedInterval {
                start,
                end_exclusive,
            } => write!(
                f,
                "interval must be non-empty and increasing: {}..{}",
                start.0, end_exclusive.0
            ),
        }
    }
}

impl std::error::Error for TimeError {}

impl TimeInterval {
    pub fn new(start: Tick, end_exclusive: Tick) -> Result<Self, TimeError> {
        if start >= end_exclusive {
            return Err(TimeError::EmptyOrReversedInterval {
                start,
                end_exclusive,
            });
        }
        Ok(Self {
            start,
            end_exclusive,
        })
    }

    pub fn start(self) -> Tick {
        self.start
    }

    pub fn end_exclusive(self) -> Tick {
        self.end_exclusive
    }

    pub fn duration_ticks(self) -> u64 {
        self.end_exclusive.0 - self.start.0
    }

    pub fn contains(self, tick: Tick) -> bool {
        self.start <= tick && tick < self.end_exclusive
    }
}

/// Event timing distinguishes instantaneous impulses/evaluations from non-empty
/// intervals; zero-length intervals are therefore never overloaded to mean both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTiming {
    Instant(Tick),
    Interval(TimeInterval),
}

impl EventTiming {
    pub fn active_at(self, tick: Tick) -> bool {
        match self {
            Self::Instant(at) => at == tick,
            Self::Interval(interval) => interval.contains(tick),
        }
    }
}
