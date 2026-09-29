//! Experimental dimension-generic packed binary hypervectors.
//!
//! This module deliberately does not replace `BinaryHV`. It provides a
//! stable research surface for comparing 1K..64K resolutions before changing
//! the production 16K ABI.

/// Supported power-of-two HDC resolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdcResolution { D1K, D2K, D4K, D8K, D16K, D32K, D64K }

impl HdcResolution {
    pub const ALL: [Self; 7] = [Self::D1K, Self::D2K, Self::D4K, Self::D8K, Self::D16K, Self::D32K, Self::D64K];
    pub const fn bits(self) -> usize { match self { Self::D1K=>1024, Self::D2K=>2048, Self::D4K=>4096, Self::D8K=>8192, Self::D16K=>16384, Self::D32K=>32768, Self::D64K=>65536 } }
    pub const fn bytes(self) -> usize { self.bits() / 8 }
}

/// A packed binary HV with an explicit resolution tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackedBinaryHv { resolution: HdcResolution, bytes: Vec<u8> }

impl PackedBinaryHv {
    pub fn zero(resolution: HdcResolution) -> Self { Self { resolution, bytes: vec![0; resolution.bytes()] } }
    pub fn resolution(&self) -> HdcResolution { self.resolution }
    pub fn as_bytes(&self) -> &[u8] { &self.bytes }
    pub fn from_bytes(resolution: HdcResolution, bytes: Vec<u8>) -> Result<Self, &'static str> {
        if bytes.len() != resolution.bytes() { return Err("packed HV byte length does not match resolution") }
        Ok(Self { resolution, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ladder_is_complete_and_monotonic() {
        let bits: Vec<_> = HdcResolution::ALL.iter().map(|r| r.bits()).collect();
        assert_eq!(bits, vec![1024,2048,4096,8192,16384,32768,65536]);
    }
    #[test] fn storage_sizes_are_exact() {
        for r in HdcResolution::ALL { assert_eq!(PackedBinaryHv::zero(r).as_bytes().len(), r.bytes()); }
    }
    #[test] fn malformed_lengths_fail_closed() {
        for r in HdcResolution::ALL { assert!(PackedBinaryHv::from_bytes(r, vec![0; r.bytes()-1]).is_err()); }
    }
}