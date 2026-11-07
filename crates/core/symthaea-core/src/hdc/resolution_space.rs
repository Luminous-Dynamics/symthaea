//! Validated, extensible HDC resolution metadata.
//!
//! This type deliberately separates an open resolution space from the finite
//! empirical ladder used by current research.

use std::{fmt, num::NonZeroUsize};

/// Evidence-backed dimensions currently used as the canonical comparison ladder.
pub const CANONICAL_RESOLUTIONS: &[usize] =
    &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];

/// Dimensions currently admitted for exploratory >64K experiments.
pub const EXPLORATORY_RESOLUTIONS: &[usize] = &[131_072, 262_144];

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HdcResolution(NonZeroUsize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionError {
    Zero,
    NotPowerOfTwo(usize),
    ByteSizeOverflow { dimensions: usize, element_size: usize },
    NotByteAligned(usize),
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "HDC resolution must be non-zero"),
            Self::NotPowerOfTwo(dim) => write!(f, "HDC resolution {dim} is not a power of two"),
            Self::ByteSizeOverflow { dimensions, element_size } =>
                write!(f, "byte size overflow for {dimensions} dimensions at {element_size}-byte elements"),
            Self::NotByteAligned(dim) => write!(f, "binary HDC resolution {dim} is not byte-aligned"),
        }
    }
}

impl std::error::Error for ResolutionError {}

impl fmt::Debug for HdcResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("HdcResolution").field(&self.dimensions()).finish()
    }
}

impl HdcResolution {
    /// Construct a validated positive power-of-two resolution.
    ///
    /// There is intentionally no 64K/256K ceiling here. Resource limits belong
    /// to allocation and execution policy, not to the representation type.
    pub const fn new(dimensions: usize) -> Result<Self, ResolutionError> {
        if dimensions == 0 {
            return Err(ResolutionError::Zero);
        }
        if !dimensions.is_power_of_two() {
            return Err(ResolutionError::NotPowerOfTwo(dimensions));
        }
        match NonZeroUsize::new(dimensions) {
            Some(value) => Ok(Self(value)),
            None => Err(ResolutionError::Zero),
        }
    }

    pub const fn dimensions(self) -> usize {
        self.0.get()
    }

    pub const fn is_canonical(self) -> bool {
        matches!(
            self.dimensions(),
            1_024 | 2_048 | 4_096 | 8_192 | 16_384 | 32_768 | 65_536
        )
    }

    pub const fn is_exploratory(self) -> bool {
        matches!(self.dimensions(), 131_072 | 262_144)
    }

    pub const fn class(self) -> ResolutionClass {
        if self.is_canonical() {
            ResolutionClass::Canonical
        } else if self.is_exploratory() {
            ResolutionClass::Exploratory
        } else {
            ResolutionClass::Custom
        }
    }

    pub const fn checked_bytes(self, element_size: usize) -> Result<usize, ResolutionError> {
        match self.dimensions().checked_mul(element_size) {
            Some(bytes) => Ok(bytes),
            None => Err(ResolutionError::ByteSizeOverflow {
                dimensions: self.dimensions(),
                element_size,
            }),
        }
    }

    pub const fn f32_bytes(self) -> Result<usize, ResolutionError> {
        self.checked_bytes(std::mem::size_of::<f32>())
    }

    pub const fn binary_bytes(self) -> Result<usize, ResolutionError> {
        if self.dimensions() % 8 != 0 {
            return Err(ResolutionError::NotByteAligned(self.dimensions()));
        }
        Ok(self.dimensions() / 8)
    }
}

impl TryFrom<usize> for HdcResolution {
    type Error = ResolutionError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionClass {
    Canonical,
    Exploratory,
    Custom,
}

pub const fn canonical_resolutions() -> &'static [usize] {
    CANONICAL_RESOLUTIONS
}

pub const fn exploratory_resolutions() -> &'static [usize] {
    EXPLORATORY_RESOLUTIONS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_128k_256k_and_future_power_of_two_tiers() {
        for &dim in &[65_536, 131_072, 262_144, 524_288, 1_048_576] {
            assert_eq!(HdcResolution::new(dim).unwrap().dimensions(), dim);
        }
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert_eq!(HdcResolution::new(0), Err(ResolutionError::Zero));
        assert_eq!(
            HdcResolution::new(100_000),
            Err(ResolutionError::NotPowerOfTwo(100_000))
        );
    }

    #[test]
    fn evidence_status_is_explicit() {
        assert_eq!(
            HdcResolution::new(65_536).unwrap().class(),
            ResolutionClass::Canonical
        );
        assert_eq!(
            HdcResolution::new(131_072).unwrap().class(),
            ResolutionClass::Exploratory
        );
        assert_eq!(
            HdcResolution::new(524_288).unwrap().class(),
            ResolutionClass::Custom
        );
    }

    #[test]
    fn binary_alignment_error_is_distinct() {
        let resolution = HdcResolution::new(4).unwrap();
        assert_eq!(
            resolution.binary_bytes(),
            Err(ResolutionError::NotByteAligned(4))
        );
    }

    #[test]
    fn byte_size_overflow_is_reported() {
        let resolution = HdcResolution::new(1usize << (usize::BITS - 1)).unwrap();
        assert_eq!(
            resolution.checked_bytes(usize::MAX),
            Err(ResolutionError::ByteSizeOverflow {
                dimensions: resolution.dimensions(),
                element_size: usize::MAX,
            })
        );
    }

    #[test]
    fn extended_memory_sizes_are_exact() {
        let d64 = HdcResolution::new(65_536).unwrap();
        let d128 = HdcResolution::new(131_072).unwrap();
        let d256 = HdcResolution::new(262_144).unwrap();

        assert_eq!(d64.f32_bytes().unwrap(), 256 * 1024);
        assert_eq!(d128.f32_bytes().unwrap(), 512 * 1024);
        assert_eq!(d256.f32_bytes().unwrap(), 1024 * 1024);
        assert_eq!(d64.binary_bytes().unwrap(), 8 * 1024);
        assert_eq!(d128.binary_bytes().unwrap(), 16 * 1024);
        assert_eq!(d256.binary_bytes().unwrap(), 32 * 1024);
    }
}
