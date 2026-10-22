//! Experimental packed-binary resolution projections.
//!
//! This is a research boundary, not a replacement for production BinaryHV.
//! Projection metadata is explicit so a lower-resolution vector is never
//! silently interpreted as a 16K vector.

use super::adaptive_resolution::{HdcResolution, PackedBinaryHv};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionFamily {
    Prefix,
    FoldXor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionSpec {
    pub family: ProjectionFamily,
    pub source: HdcResolution,
    pub target: HdcResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionError {
    SameResolution,
    ExpansionUnsupported,
    NonDivisibleFold,
    InvalidSource,
}

impl ProjectionSpec {
    pub const fn new(family: ProjectionFamily, source: HdcResolution, target: HdcResolution) -> Self {
        Self { family, source, target }
    }

    pub fn demote(&self, input: &PackedBinaryHv) -> Result<PackedBinaryHv, ProjectionError> {
        if input.resolution() != self.source {
            return Err(ProjectionError::InvalidSource);
        }
        if self.source.bits() == self.target.bits() {
            return Err(ProjectionError::SameResolution);
        }
        if self.target.bits() > self.source.bits() {
            return Err(ProjectionError::ExpansionUnsupported);
        }

        match self.family {
            ProjectionFamily::Prefix => {
                let n = self.target.bytes();
                PackedBinaryHv::from_bytes(self.target, input.as_bytes()[..n].to_vec())
                    .map_err(|_| ProjectionError::InvalidSource)
            }
            ProjectionFamily::FoldXor => {
                let ratio = self.source.bits() / self.target.bits();
                if ratio == 0 || !ratio.is_power_of_two() {
                    return Err(ProjectionError::NonDivisibleFold);
                }
                let mut out = vec![0u8; self.target.bytes()];
                for chunk in input.as_bytes().chunks_exact(self.target.bytes()) {
                    for (dst, src) in out.iter_mut().zip(chunk) {
                        *dst ^= *src;
                    }
                }
                PackedBinaryHv::from_bytes(self.target, out)
                    .map_err(|_| ProjectionError::InvalidSource)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(resolution: HdcResolution) -> PackedBinaryHv {
        let bytes: Vec<u8> = (0..resolution.bytes())
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();
        PackedBinaryHv::from_bytes(resolution, bytes).unwrap()
    }

    #[test]
    fn projection_is_deterministic() {
        let input = fixture(HdcResolution::D64K);
        let spec = ProjectionSpec::new(ProjectionFamily::FoldXor, HdcResolution::D64K, HdcResolution::D4K);
        assert_eq!(spec.demote(&input), spec.demote(&input));
    }

    #[test]
    fn prefix_preserves_exact_prefix() {
        let input = fixture(HdcResolution::D16K);
        let spec = ProjectionSpec::new(ProjectionFamily::Prefix, HdcResolution::D16K, HdcResolution::D4K);
        let output = spec.demote(&input).unwrap();
        assert_eq!(output.as_bytes(), &input.as_bytes()[..HdcResolution::D4K.bytes()]);
    }

    #[test]
    fn fold_xor_is_dimension_exact() {
        let input = fixture(HdcResolution::D16K);
        let spec = ProjectionSpec::new(ProjectionFamily::FoldXor, HdcResolution::D16K, HdcResolution::D4K);
        let output = spec.demote(&input).unwrap();
        assert_eq!(output.resolution(), HdcResolution::D4K);
        assert_eq!(output.as_bytes().len(), HdcResolution::D4K.bytes());
    }

    #[test]
    fn wrong_source_fails_closed() {
        let input = fixture(HdcResolution::D16K);
        let spec = ProjectionSpec::new(ProjectionFamily::FoldXor, HdcResolution::D64K, HdcResolution::D4K);
        assert_eq!(spec.demote(&input), Err(ProjectionError::InvalidSource));
    }

    #[test]
    fn expansion_fails_closed() {
        let input = fixture(HdcResolution::D4K);
        let spec = ProjectionSpec::new(ProjectionFamily::FoldXor, HdcResolution::D4K, HdcResolution::D16K);
        assert_eq!(spec.demote(&input), Err(ProjectionError::ExpansionUnsupported));
    }
}
