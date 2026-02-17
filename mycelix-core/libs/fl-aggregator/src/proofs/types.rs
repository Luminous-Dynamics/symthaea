//! Common Proof Types
//!
//! Shared types for proof generation and verification across all circuits.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Proof circuit types supported by this module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofType {
    /// Range proof - prove value is within bounds
    Range,
    /// Merkle membership proof - prove element exists in tree
    Membership,
    /// Gradient integrity proof - prove valid FL contribution
    GradientIntegrity,
    /// Identity assurance proof - prove assurance level threshold
    IdentityAssurance,
    /// Vote eligibility proof - prove voter qualification
    VoteEligibility,
}

impl ProofType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ProofType::Range => "RangeProof",
            ProofType::Membership => "MembershipProof",
            ProofType::GradientIntegrity => "GradientIntegrityProof",
            ProofType::IdentityAssurance => "IdentityAssuranceProof",
            ProofType::VoteEligibility => "VoteEligibilityProof",
        }
    }
}

/// Security level for proof generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 96-bit security (faster, smaller proofs)
    Standard96,
    /// 128-bit security (default, recommended)
    #[default]
    Standard128,
    /// 256-bit security (maximum security, larger proofs)
    High256,
}

impl SecurityLevel {
    /// Get the number of queries for FRI protocol
    pub fn num_queries(&self) -> usize {
        match self {
            SecurityLevel::Standard96 => 27,
            SecurityLevel::Standard128 => 36,
            SecurityLevel::High256 => 72,
        }
    }

    /// Get the blowup factor
    pub fn blowup_factor(&self) -> usize {
        match self {
            SecurityLevel::Standard96 => 8,
            SecurityLevel::Standard128 => 8,
            SecurityLevel::High256 => 16,
        }
    }

    /// Get grinding factor (proof-of-work bits)
    pub fn grinding_factor(&self) -> u32 {
        match self {
            SecurityLevel::Standard96 => 16,
            SecurityLevel::Standard128 => 20,
            SecurityLevel::High256 => 24,
        }
    }
}

/// Configuration for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofConfig {
    /// Security level
    pub security_level: SecurityLevel,

    /// Enable parallel proof generation
    pub parallel: bool,

    /// Maximum proof size in bytes (0 = unlimited)
    pub max_proof_size: usize,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::default(),
            parallel: true,
            max_proof_size: 0,
        }
    }
}

/// Result of proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is valid
    pub valid: bool,

    /// Proof type that was verified
    pub proof_type: ProofType,

    /// Verification time
    pub verification_time: Duration,

    /// Additional verification details
    pub details: Option<String>,
}

impl VerificationResult {
    /// Create a successful verification result
    pub fn success(proof_type: ProofType, time: Duration) -> Self {
        Self {
            valid: true,
            proof_type,
            verification_time: time,
            details: None,
        }
    }

    /// Create a failed verification result
    pub fn failure(proof_type: ProofType, time: Duration, reason: impl Into<String>) -> Self {
        Self {
            valid: false,
            proof_type,
            verification_time: time,
            details: Some(reason.into()),
        }
    }
}

/// Statistics about proof generation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofStats {
    /// Number of proofs generated
    pub proofs_generated: u64,

    /// Number of proofs verified
    pub proofs_verified: u64,

    /// Number of verification failures
    pub verification_failures: u64,

    /// Total proof generation time (milliseconds)
    pub total_generation_time_ms: u64,

    /// Total verification time (milliseconds)
    pub total_verification_time_ms: u64,

    /// Average proof size in bytes
    pub average_proof_size: usize,
}

impl ProofStats {
    /// Record a proof generation
    pub fn record_generation(&mut self, duration: Duration, size: usize) {
        self.proofs_generated += 1;
        self.total_generation_time_ms += duration.as_millis() as u64;

        // Update running average
        let total_size = self.average_proof_size as u64 * (self.proofs_generated - 1) + size as u64;
        self.average_proof_size = (total_size / self.proofs_generated) as usize;
    }

    /// Record a proof verification
    pub fn record_verification(&mut self, duration: Duration, valid: bool) {
        self.proofs_verified += 1;
        self.total_verification_time_ms += duration.as_millis() as u64;
        if !valid {
            self.verification_failures += 1;
        }
    }

    /// Get average generation time in milliseconds
    pub fn avg_generation_time_ms(&self) -> u64 {
        if self.proofs_generated > 0 {
            self.total_generation_time_ms / self.proofs_generated
        } else {
            0
        }
    }

    /// Get average verification time in milliseconds
    pub fn avg_verification_time_ms(&self) -> u64 {
        if self.proofs_verified > 0 {
            self.total_verification_time_ms / self.proofs_verified
        } else {
            0
        }
    }
}

/// Commitment to a value (hash-based)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    /// The commitment value (256-bit hash)
    pub value: [u8; 32],
}

impl Commitment {
    /// Create a new commitment from bytes
    pub fn new(value: [u8; 32]) -> Self {
        Self { value }
    }

    /// Create from a slice (must be 32 bytes)
    pub fn from_slice(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 32 {
            let mut value = [0u8; 32];
            value.copy_from_slice(bytes);
            Some(Self { value })
        } else {
            None
        }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.value
    }
}

// ============================================================================
// Safe Byte Parsing Utilities
// ============================================================================

/// A safe byte reader that tracks position and validates bounds
#[derive(Debug)]
pub struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    /// Create a new byte reader
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.offset)
    }

    /// Check if there are at least n bytes remaining
    pub fn has_remaining(&self, n: usize) -> bool {
        self.remaining() >= n
    }

    /// Get current offset
    pub fn position(&self) -> usize {
        self.offset
    }

    /// Read a u8
    pub fn read_u8(&mut self) -> Option<u8> {
        if self.offset < self.data.len() {
            let val = self.data[self.offset];
            self.offset += 1;
            Some(val)
        } else {
            None
        }
    }

    /// Read a u16 (little-endian)
    pub fn read_u16_le(&mut self) -> Option<u16> {
        if self.offset + 2 <= self.data.len() {
            let bytes: [u8; 2] = self.data[self.offset..self.offset + 2]
                .try_into()
                .ok()?;
            self.offset += 2;
            Some(u16::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Read a u32 (little-endian)
    pub fn read_u32_le(&mut self) -> Option<u32> {
        if self.offset + 4 <= self.data.len() {
            let bytes: [u8; 4] = self.data[self.offset..self.offset + 4]
                .try_into()
                .ok()?;
            self.offset += 4;
            Some(u32::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Read a u64 (little-endian)
    pub fn read_u64_le(&mut self) -> Option<u64> {
        if self.offset + 8 <= self.data.len() {
            let bytes: [u8; 8] = self.data[self.offset..self.offset + 8]
                .try_into()
                .ok()?;
            self.offset += 8;
            Some(u64::from_le_bytes(bytes))
        } else {
            None
        }
    }

    /// Read exactly n bytes
    pub fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.offset + n <= self.data.len() {
            let slice = &self.data[self.offset..self.offset + n];
            self.offset += n;
            Some(slice)
        } else {
            None
        }
    }

    /// Read exactly 32 bytes as an array
    pub fn read_32_bytes(&mut self) -> Option<[u8; 32]> {
        self.read_bytes(32).and_then(|s| s.try_into().ok())
    }

    /// Read a length-prefixed string (u16 length prefix)
    pub fn read_string_u16(&mut self) -> Option<String> {
        let len = self.read_u16_le()? as usize;
        if len == 0 {
            return Some(String::new());
        }
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).ok()
    }

    /// Read remaining bytes
    pub fn read_remaining(&mut self) -> &'a [u8] {
        let slice = &self.data[self.offset..];
        self.offset = self.data.len();
        slice
    }

    /// Skip n bytes
    pub fn skip(&mut self, n: usize) -> bool {
        if self.offset + n <= self.data.len() {
            self.offset += n;
            true
        } else {
            false
        }
    }
}

/// A safe byte writer for building serialized data
#[derive(Debug, Default)]
pub struct ByteWriter {
    data: Vec<u8>,
}

impl ByteWriter {
    /// Create a new byte writer
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Write a u8
    pub fn write_u8(&mut self, val: u8) {
        self.data.push(val);
    }

    /// Write a u16 (little-endian)
    pub fn write_u16_le(&mut self, val: u16) {
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    /// Write a u32 (little-endian)
    pub fn write_u32_le(&mut self, val: u32) {
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    /// Write a u64 (little-endian)
    pub fn write_u64_le(&mut self, val: u64) {
        self.data.extend_from_slice(&val.to_le_bytes());
    }

    /// Write bytes
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    /// Write a length-prefixed string (u16 length prefix)
    pub fn write_string_u16(&mut self, s: &str) {
        let len = s.len().min(u16::MAX as usize) as u16;
        self.write_u16_le(len);
        self.write_bytes(&s.as_bytes()[..len as usize]);
    }

    /// Get the written data
    pub fn finish(self) -> Vec<u8> {
        self.data
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_type_names() {
        assert_eq!(ProofType::Range.name(), "RangeProof");
        assert_eq!(ProofType::Membership.name(), "MembershipProof");
        assert_eq!(ProofType::GradientIntegrity.name(), "GradientIntegrityProof");
    }

    #[test]
    fn test_security_levels() {
        assert!(SecurityLevel::Standard128.num_queries() > SecurityLevel::Standard96.num_queries());
        assert!(SecurityLevel::High256.num_queries() > SecurityLevel::Standard128.num_queries());
    }

    #[test]
    fn test_verification_result() {
        let success = VerificationResult::success(ProofType::Range, Duration::from_millis(10));
        assert!(success.valid);

        let failure = VerificationResult::failure(
            ProofType::Range,
            Duration::from_millis(5),
            "constraint failed",
        );
        assert!(!failure.valid);
        assert!(failure.details.is_some());
    }

    #[test]
    fn test_proof_stats() {
        let mut stats = ProofStats::default();

        stats.record_generation(Duration::from_millis(100), 1000);
        stats.record_generation(Duration::from_millis(200), 2000);

        assert_eq!(stats.proofs_generated, 2);
        assert_eq!(stats.avg_generation_time_ms(), 150);
        assert_eq!(stats.average_proof_size, 1500);
    }

    #[test]
    fn test_commitment() {
        let bytes = [42u8; 32];
        let commitment = Commitment::new(bytes);
        assert_eq!(commitment.as_bytes(), &bytes);

        let from_slice = Commitment::from_slice(&bytes).unwrap();
        assert_eq!(commitment, from_slice);

        assert!(Commitment::from_slice(&[0u8; 16]).is_none());
    }

    #[test]
    fn test_byte_reader_basic() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut reader = ByteReader::new(&data);

        assert_eq!(reader.remaining(), 8);
        assert_eq!(reader.read_u8(), Some(1));
        assert_eq!(reader.remaining(), 7);
        assert_eq!(reader.read_u8(), Some(2));
        assert_eq!(reader.position(), 2);
    }

    #[test]
    fn test_byte_reader_integers() {
        let mut writer = ByteWriter::new();
        writer.write_u16_le(0x1234);
        writer.write_u32_le(0x56789ABC);
        writer.write_u64_le(0xDEADBEEFCAFEBABE);

        let data = writer.finish();
        let mut reader = ByteReader::new(&data);

        assert_eq!(reader.read_u16_le(), Some(0x1234));
        assert_eq!(reader.read_u32_le(), Some(0x56789ABC));
        assert_eq!(reader.read_u64_le(), Some(0xDEADBEEFCAFEBABE));
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn test_byte_reader_bounds() {
        let data = [1u8, 2, 3];
        let mut reader = ByteReader::new(&data);

        // Try to read 4 bytes when only 3 available
        assert!(reader.read_u32_le().is_none());
        // Reader should not advance on failure
        assert_eq!(reader.position(), 0);

        // Read what's available
        assert_eq!(reader.read_u8(), Some(1));
        assert_eq!(reader.read_u8(), Some(2));
        assert_eq!(reader.read_u8(), Some(3));
        assert!(reader.read_u8().is_none());
    }

    #[test]
    fn test_byte_reader_32_bytes() {
        let mut data = vec![0u8; 40];
        for i in 0..32 {
            data[i] = i as u8;
        }

        let mut reader = ByteReader::new(&data);
        let arr = reader.read_32_bytes().unwrap();

        for i in 0..32 {
            assert_eq!(arr[i], i as u8);
        }
        assert_eq!(reader.remaining(), 8);
    }

    #[test]
    fn test_byte_reader_string() {
        let mut writer = ByteWriter::new();
        writer.write_string_u16("hello");
        writer.write_string_u16("");
        writer.write_string_u16("world");

        let data = writer.finish();
        let mut reader = ByteReader::new(&data);

        assert_eq!(reader.read_string_u16(), Some("hello".to_string()));
        assert_eq!(reader.read_string_u16(), Some("".to_string()));
        assert_eq!(reader.read_string_u16(), Some("world".to_string()));
    }

    #[test]
    fn test_byte_writer() {
        let mut writer = ByteWriter::with_capacity(100);

        writer.write_u8(42);
        writer.write_bytes(&[1, 2, 3]);

        assert_eq!(writer.len(), 4);
        assert!(!writer.is_empty());

        let data = writer.finish();
        assert_eq!(data, vec![42, 1, 2, 3]);
    }
}
