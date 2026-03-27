// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! External timestamp anchoring for Merkle roots.
//!
//! Provides a trait-based system for anchoring [`MerkleTimestampRoot`] hashes
//! to external systems (local files, git repositories, etc.), creating
//! independently verifiable proof that a merkle root existed at a given time.
//!
//! ## Architecture
//!
//! - [`TimestampAnchor`] — trait that any backend must implement
//! - [`LocalFileAnchor`] — writes JSON anchor files to a directory
//! - [`GitAnchor`] — creates git tags containing the root hash
//! - [`MultiAnchor`] — fans out to multiple backends for redundancy
//! - [`AnchorChain`] — ring buffer tracking receipts over time

use serde::{Deserialize, Serialize};

use crate::merkle_timestamp::MerkleTimestampRoot;

// ============================================================================
// Types
// ============================================================================

/// Result of anchoring a merkle root to an external system.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnchorReceipt {
    /// The root hash that was anchored.
    pub root_hash: [u8; 32],
    /// Which backend produced this receipt.
    pub anchor_type: AnchorType,
    /// Backend-specific identifier (file path, git tag, tx hash, etc.).
    pub anchor_id: String,
    /// Timestamp when the anchoring occurred (microseconds since epoch).
    pub anchored_at: u64,
    /// Optional URL where the anchor can be independently verified.
    pub verification_url: Option<String>,
}

/// Identifies the external system used for anchoring.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnchorType {
    /// Written to a local timestamped JSON file.
    LocalFile,
    /// Committed as a lightweight git tag.
    GitCommit,
    /// RFC 3161 timestamp authority (future).
    HttpTimestamp,
    /// Bitcoin OP_RETURN (future).
    Bitcoin,
    /// Ethereum calldata (future).
    Ethereum,
    /// User-defined backend.
    Custom(String),
}

/// Errors that can occur during anchoring or verification.
#[derive(Clone, Debug)]
pub enum AnchorError {
    /// File system I/O failure.
    IoError(String),
    /// Network request failure.
    NetworkError(String),
    /// Verification of an existing anchor failed.
    VerificationFailed(String),
    /// The requested operation is not supported by this backend.
    NotSupported(String),
}

impl core::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::NetworkError(msg) => write!(f, "Network error: {msg}"),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {msg}"),
            Self::NotSupported(msg) => write!(f, "Not supported: {msg}"),
        }
    }
}

// ============================================================================
// Trait
// ============================================================================

/// Trait for external timestamp anchoring backends.
///
/// Implementors anchor a [`MerkleTimestampRoot`] to an external system and
/// return an [`AnchorReceipt`] that can later be used for verification.
pub trait TimestampAnchor {
    /// Anchor a merkle root to the external system.
    fn anchor(&self, root: &MerkleTimestampRoot) -> Result<AnchorReceipt, AnchorError>;

    /// Verify a previously created anchor receipt.
    fn verify(&self, receipt: &AnchorReceipt) -> Result<bool, AnchorError>;

    /// The anchor type this backend produces.
    fn anchor_type(&self) -> AnchorType;
}

// ============================================================================
// Helpers (no serde_json at runtime — manual JSON formatting)
// ============================================================================

/// Format a `[u8; 32]` as a lowercase hex string.
fn hex_encode(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a 64-char hex string into `[u8; 32]`.  Returns `None` on bad input.
fn hex_decode(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Format an anchor file's contents as JSON without depending on serde_json
/// at runtime (it is dev-only in this crate).
fn format_anchor_json(root: &MerkleTimestampRoot) -> String {
    let prev = match &root.previous_root {
        Some(h) => format!("\"{}\"", hex_encode(h)),
        None => "null".to_string(),
    };
    format!(
        "{{\n  \"root_hash\": \"{}\",\n  \"leaf_count\": {},\n  \"created_at\": {},\n  \"sequence_number\": {},\n  \"previous_root\": {}\n}}",
        hex_encode(&root.root_hash),
        root.leaf_count,
        root.created_at,
        root.sequence_number,
        prev,
    )
}

/// Extract the `root_hash` hex string from a manually-formatted JSON file.
/// Returns the decoded hash, or `None` if the file doesn't match.
fn extract_root_hash_from_json(content: &str) -> Option<[u8; 32]> {
    // Look for `"root_hash": "HEX"`
    let marker = "\"root_hash\": \"";
    let start = content.find(marker)? + marker.len();
    let end = start + 64;
    if content.len() < end {
        return None;
    }
    hex_decode(&content[start..end])
}

// ============================================================================
// LocalFileAnchor
// ============================================================================

/// Anchors merkle roots by writing timestamped JSON files to a directory.
///
/// Each file is named `anchor_{sequence}_{timestamp}.json` and contains the
/// root hash plus metadata.  Verification reads the file back and checks that
/// the stored root hash matches the receipt.
pub struct LocalFileAnchor {
    directory: String,
}

impl LocalFileAnchor {
    /// Create a new file anchor writing to `directory`.
    ///
    /// The directory is created on first `anchor()` call if it doesn't exist.
    pub fn new(directory: &str) -> Self {
        Self {
            directory: directory.to_string(),
        }
    }

    /// Build the filename for a given root.
    fn filename(&self, root: &MerkleTimestampRoot) -> String {
        format!(
            "{}/anchor_{}_{}.json",
            self.directory, root.sequence_number, root.created_at
        )
    }
}

impl TimestampAnchor for LocalFileAnchor {
    fn anchor(&self, root: &MerkleTimestampRoot) -> Result<AnchorReceipt, AnchorError> {
        // Ensure directory exists.
        std::fs::create_dir_all(&self.directory)
            .map_err(|e| AnchorError::IoError(format!("create_dir_all: {e}")))?;

        let path = self.filename(root);
        let json = format_anchor_json(root);
        std::fs::write(&path, json)
            .map_err(|e| AnchorError::IoError(format!("write {path}: {e}")))?;

        // Use created_at from the root as the anchor timestamp.
        Ok(AnchorReceipt {
            root_hash: root.root_hash,
            anchor_type: AnchorType::LocalFile,
            anchor_id: path,
            anchored_at: root.created_at,
            verification_url: None,
        })
    }

    fn verify(&self, receipt: &AnchorReceipt) -> Result<bool, AnchorError> {
        let content = std::fs::read_to_string(&receipt.anchor_id)
            .map_err(|e| AnchorError::IoError(format!("read {}: {e}", receipt.anchor_id)))?;

        match extract_root_hash_from_json(&content) {
            Some(stored_hash) => Ok(stored_hash == receipt.root_hash),
            None => Ok(false),
        }
    }

    fn anchor_type(&self) -> AnchorType {
        AnchorType::LocalFile
    }
}

// ============================================================================
// GitAnchor
// ============================================================================

/// Anchors merkle roots by creating lightweight git tags.
///
/// Each tag is named `merkle-root-{sequence}` with a message containing the
/// hex-encoded root hash and metadata.  Uses `std::process::Command` to
/// invoke `git`.
pub struct GitAnchor {
    repo_path: String,
}

impl GitAnchor {
    /// Create a new git anchor operating on the repository at `repo_path`.
    pub fn new(repo_path: &str) -> Self {
        Self {
            repo_path: repo_path.to_string(),
        }
    }

    /// Build the tag name for a given sequence number.
    fn tag_name(sequence: u64) -> String {
        format!("merkle-root-{sequence}")
    }
}

impl TimestampAnchor for GitAnchor {
    fn anchor(&self, root: &MerkleTimestampRoot) -> Result<AnchorReceipt, AnchorError> {
        let tag = Self::tag_name(root.sequence_number);
        let message = format!(
            "root_hash={}\nleaf_count={}\ncreated_at={}\nsequence={}",
            hex_encode(&root.root_hash),
            root.leaf_count,
            root.created_at,
            root.sequence_number,
        );

        let output = std::process::Command::new("git")
            .args(["tag", "-a", &tag, "-m", &message])
            .current_dir(&self.repo_path)
            .output()
            .map_err(|e| AnchorError::IoError(format!("git tag: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AnchorError::IoError(format!("git tag failed: {stderr}")));
        }

        Ok(AnchorReceipt {
            root_hash: root.root_hash,
            anchor_type: AnchorType::GitCommit,
            anchor_id: tag,
            anchored_at: root.created_at,
            verification_url: None,
        })
    }

    fn verify(&self, receipt: &AnchorReceipt) -> Result<bool, AnchorError> {
        let output = std::process::Command::new("git")
            .args(["tag", "-n1", &receipt.anchor_id])
            .current_dir(&self.repo_path)
            .output()
            .map_err(|e| AnchorError::IoError(format!("git tag -n1: {e}")))?;

        if !output.status.success() {
            return Ok(false);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let expected_hex = hex_encode(&receipt.root_hash);
        Ok(stdout.contains(&expected_hex))
    }

    fn anchor_type(&self) -> AnchorType {
        AnchorType::GitCommit
    }
}

// ============================================================================
// MultiAnchor
// ============================================================================

/// Fans out anchoring to multiple backends for redundancy.
///
/// Each backend is tried independently; failures in one do not block others.
pub struct MultiAnchor {
    anchors: Vec<Box<dyn TimestampAnchor>>,
}

impl MultiAnchor {
    /// Create an empty multi-anchor.
    pub fn new() -> Self {
        Self {
            anchors: Vec::new(),
        }
    }

    /// Add a backend.
    pub fn add_anchor(&mut self, anchor: Box<dyn TimestampAnchor>) {
        self.anchors.push(anchor);
    }

    /// Anchor to all backends, returning one result per backend.
    pub fn anchor_all(
        &self,
        root: &MerkleTimestampRoot,
    ) -> Vec<Result<AnchorReceipt, AnchorError>> {
        self.anchors.iter().map(|a| a.anchor(root)).collect()
    }
}

impl Default for MultiAnchor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AnchorChain
// ============================================================================

/// Ring buffer of [`AnchorReceipt`]s linking merkle roots to their external
/// anchors over time.
pub struct AnchorChain {
    receipts: Vec<AnchorReceipt>,
    max_receipts: usize,
}

/// Default ring buffer capacity.
const DEFAULT_ANCHOR_CHAIN_CAPACITY: usize = 1024;

impl AnchorChain {
    /// Create a chain with the default capacity (1024).
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
            max_receipts: DEFAULT_ANCHOR_CHAIN_CAPACITY,
        }
    }

    /// Create a chain with a custom capacity.
    pub fn with_capacity(max: usize) -> Self {
        Self {
            receipts: Vec::new(),
            max_receipts: max,
        }
    }

    /// Add a receipt, evicting the oldest if at capacity.
    pub fn add_receipt(&mut self, receipt: AnchorReceipt) {
        if self.receipts.len() >= self.max_receipts {
            self.receipts.remove(0);
        }
        self.receipts.push(receipt);
    }

    /// Find a receipt by its root hash.
    pub fn find_by_root(&self, root_hash: &[u8; 32]) -> Option<&AnchorReceipt> {
        self.receipts.iter().find(|r| &r.root_hash == root_hash)
    }

    /// Find a receipt by its anchor ID (file path, tag name, etc.).
    pub fn find_by_anchor_id(&self, anchor_id: &str) -> Option<&AnchorReceipt> {
        self.receipts.iter().find(|r| r.anchor_id == anchor_id)
    }

    /// All stored receipts.
    pub fn all_receipts(&self) -> &[AnchorReceipt] {
        &self.receipts
    }

    /// The most recently added receipt, if any.
    pub fn latest(&self) -> Option<&AnchorReceipt> {
        self.receipts.last()
    }
}

impl Default for AnchorChain {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_timestamp::MerkleTimestampRoot;

    /// Helper: build a root with a deterministic hash for testing.
    fn make_root(seq: u64, ts: u64) -> MerkleTimestampRoot {
        let mut hash = [0u8; 32];
        hash[0] = seq as u8;
        hash[1] = (seq >> 8) as u8;
        MerkleTimestampRoot {
            root_hash: hash,
            leaf_count: 5,
            created_at: ts,
            sequence_number: seq,
            previous_root: None,
        }
    }

    /// Helper: create a temp directory that is cleaned up on drop.
    struct TempDir(String);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = format!("/tmp/mycelix_anchor_test_{name}_{}", std::process::id());
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &str {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // ---- 1. LocalFileAnchor: anchor + verify roundtrip ----

    #[test]
    fn local_file_anchor_roundtrip() {
        let dir = TempDir::new("roundtrip");
        let anchor = LocalFileAnchor::new(dir.path());
        let root = make_root(0, 1000);

        let receipt = anchor.anchor(&root).expect("anchor should succeed");
        assert_eq!(receipt.root_hash, root.root_hash);
        assert_eq!(receipt.anchor_type, AnchorType::LocalFile);
        assert!(receipt.anchor_id.contains("anchor_0_1000.json"));
        assert_eq!(receipt.anchored_at, 1000);

        let verified = anchor.verify(&receipt).expect("verify should succeed");
        assert!(verified, "receipt should verify against the file");
    }

    // ---- 2. LocalFileAnchor: verify fails with wrong root hash ----

    #[test]
    fn local_file_anchor_verify_fails_wrong_hash() {
        let dir = TempDir::new("wrong_hash");
        let anchor = LocalFileAnchor::new(dir.path());
        let root = make_root(0, 2000);

        let mut receipt = anchor.anchor(&root).expect("anchor");
        // Tamper with the root hash in the receipt.
        receipt.root_hash[0] ^= 0xFF;

        let verified = anchor.verify(&receipt).expect("verify should not error");
        assert!(!verified, "tampered hash should not verify");
    }

    // ---- 3. LocalFileAnchor: verify fails with missing file ----

    #[test]
    fn local_file_anchor_verify_fails_missing_file() {
        let dir = TempDir::new("missing");
        let anchor = LocalFileAnchor::new(dir.path());

        let receipt = AnchorReceipt {
            root_hash: [0u8; 32],
            anchor_type: AnchorType::LocalFile,
            anchor_id: format!("{}/nonexistent.json", dir.path()),
            anchored_at: 0,
            verification_url: None,
        };

        let result = anchor.verify(&receipt);
        assert!(result.is_err(), "missing file should produce IoError");
        match result.unwrap_err() {
            AnchorError::IoError(_) => {}
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ---- 4. GitAnchor: struct creation (no actual git operations) ----

    #[test]
    fn git_anchor_creation() {
        let anchor = GitAnchor::new("/tmp/fake-repo");
        assert_eq!(anchor.anchor_type(), AnchorType::GitCommit);
        assert_eq!(anchor.repo_path, "/tmp/fake-repo");
    }

    #[test]
    fn git_anchor_tag_name_format() {
        assert_eq!(GitAnchor::tag_name(0), "merkle-root-0");
        assert_eq!(GitAnchor::tag_name(42), "merkle-root-42");
        assert_eq!(GitAnchor::tag_name(999), "merkle-root-999");
    }

    // ---- 5. MultiAnchor: multiple backends, mixed results ----

    #[test]
    fn multi_anchor_mixed_results() {
        let dir1 = TempDir::new("multi1");
        let dir2 = TempDir::new("multi2");

        let mut multi = MultiAnchor::new();
        multi.add_anchor(Box::new(LocalFileAnchor::new(dir1.path())));
        multi.add_anchor(Box::new(LocalFileAnchor::new(dir2.path())));
        // Add a git anchor pointing at a nonexistent repo (will fail).
        multi.add_anchor(Box::new(GitAnchor::new("/tmp/nonexistent-repo-xyz")));

        let root = make_root(7, 5000);
        let results = multi.anchor_all(&root);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok(), "first file anchor should succeed");
        assert!(results[1].is_ok(), "second file anchor should succeed");
        assert!(results[2].is_err(), "git anchor on fake repo should fail");
    }

    // ---- 6. AnchorChain: add, find, ring buffer overflow ----

    #[test]
    fn anchor_chain_add_and_find_by_root() {
        let mut chain = AnchorChain::new();
        let root = make_root(0, 100);
        let receipt = AnchorReceipt {
            root_hash: root.root_hash,
            anchor_type: AnchorType::LocalFile,
            anchor_id: "file_0.json".to_string(),
            anchored_at: 100,
            verification_url: None,
        };

        chain.add_receipt(receipt.clone());
        assert_eq!(chain.all_receipts().len(), 1);

        let found = chain.find_by_root(&root.root_hash);
        assert!(found.is_some());
        assert_eq!(found.unwrap().anchor_id, "file_0.json");
    }

    #[test]
    fn anchor_chain_find_by_anchor_id() {
        let mut chain = AnchorChain::new();
        let receipt = AnchorReceipt {
            root_hash: [1u8; 32],
            anchor_type: AnchorType::GitCommit,
            anchor_id: "merkle-root-5".to_string(),
            anchored_at: 500,
            verification_url: None,
        };

        chain.add_receipt(receipt);
        assert!(chain.find_by_anchor_id("merkle-root-5").is_some());
        assert!(chain.find_by_anchor_id("merkle-root-99").is_none());
    }

    #[test]
    fn anchor_chain_latest() {
        let mut chain = AnchorChain::new();
        assert!(chain.latest().is_none());

        for i in 0..3u8 {
            chain.add_receipt(AnchorReceipt {
                root_hash: [i; 32],
                anchor_type: AnchorType::LocalFile,
                anchor_id: format!("file_{i}.json"),
                anchored_at: i as u64 * 100,
                verification_url: None,
            });
        }

        let latest = chain.latest().expect("should have latest");
        assert_eq!(latest.anchor_id, "file_2.json");
    }

    #[test]
    fn anchor_chain_ring_buffer_overflow() {
        let mut chain = AnchorChain::with_capacity(3);

        for i in 0..5u8 {
            chain.add_receipt(AnchorReceipt {
                root_hash: [i; 32],
                anchor_type: AnchorType::LocalFile,
                anchor_id: format!("file_{i}.json"),
                anchored_at: i as u64 * 100,
                verification_url: None,
            });
        }

        assert_eq!(chain.all_receipts().len(), 3);
        // Oldest surviving should be index 2 (0 and 1 evicted).
        assert_eq!(chain.all_receipts()[0].anchor_id, "file_2.json");
        assert_eq!(chain.all_receipts()[2].anchor_id, "file_4.json");
    }

    // ---- 7. AnchorReceipt: serde roundtrip ----

    #[test]
    fn anchor_receipt_serde_roundtrip() {
        let receipt = AnchorReceipt {
            root_hash: [42u8; 32],
            anchor_type: AnchorType::LocalFile,
            anchor_id: "/tmp/test.json".to_string(),
            anchored_at: 123456,
            verification_url: Some("https://example.com/verify".to_string()),
        };

        let json = serde_json::to_string(&receipt).expect("serialize");
        let decoded: AnchorReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt, decoded);
    }

    #[test]
    fn anchor_receipt_serde_roundtrip_no_url() {
        let receipt = AnchorReceipt {
            root_hash: [0u8; 32],
            anchor_type: AnchorType::GitCommit,
            anchor_id: "merkle-root-0".to_string(),
            anchored_at: 0,
            verification_url: None,
        };

        let json = serde_json::to_string(&receipt).expect("serialize");
        let decoded: AnchorReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt, decoded);
    }

    // ---- 8. AnchorType: serde roundtrip for each variant ----

    #[test]
    fn anchor_type_serde_roundtrip_all_variants() {
        let variants = vec![
            AnchorType::LocalFile,
            AnchorType::GitCommit,
            AnchorType::HttpTimestamp,
            AnchorType::Bitcoin,
            AnchorType::Ethereum,
            AnchorType::Custom("my-backend".to_string()),
        ];

        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let decoded: AnchorType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(variant, decoded, "roundtrip failed for {json}");
        }
    }

    // ---- 9. AnchorError: display formatting ----

    #[test]
    fn anchor_error_display() {
        let cases = vec![
            (AnchorError::IoError("disk full".into()), "IO error: disk full"),
            (
                AnchorError::NetworkError("timeout".into()),
                "Network error: timeout",
            ),
            (
                AnchorError::VerificationFailed("hash mismatch".into()),
                "Verification failed: hash mismatch",
            ),
            (
                AnchorError::NotSupported("Bitcoin".into()),
                "Not supported: Bitcoin",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(format!("{err}"), expected);
        }
    }

    // ---- 10. Hex encoding/decoding helpers ----

    #[test]
    fn hex_roundtrip() {
        let bytes = [
            0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0xFF,
        ];
        let hex = hex_encode(&bytes);
        let decoded = hex_decode(&hex).expect("should decode");
        assert_eq!(bytes, decoded);
    }

    #[test]
    fn hex_decode_rejects_bad_input() {
        assert!(hex_decode("too_short").is_none());
        assert!(hex_decode(&"g".repeat(64)).is_none()); // invalid hex char
    }

    // ---- 11. format_anchor_json produces parseable output ----

    #[test]
    fn format_anchor_json_contains_root_hash() {
        let root = make_root(3, 9000);
        let json = format_anchor_json(&root);

        // Should contain the hex-encoded root hash.
        let expected_hex = hex_encode(&root.root_hash);
        assert!(json.contains(&expected_hex));

        // extract_root_hash_from_json should round-trip.
        let extracted = extract_root_hash_from_json(&json).expect("should extract");
        assert_eq!(extracted, root.root_hash);
    }

    #[test]
    fn format_anchor_json_with_previous_root() {
        let mut root = make_root(1, 2000);
        root.previous_root = Some([0xAA; 32]);
        let json = format_anchor_json(&root);

        // Should contain the previous root hex.
        let prev_hex = hex_encode(&[0xAA; 32]);
        assert!(json.contains(&prev_hex));
    }

    // ---- 12. LocalFileAnchor writes correct file contents ----

    #[test]
    fn local_file_anchor_file_contents_are_valid() {
        let dir = TempDir::new("contents");
        let anchor = LocalFileAnchor::new(dir.path());
        let root = make_root(2, 3000);

        let receipt = anchor.anchor(&root).expect("anchor");
        let content = std::fs::read_to_string(&receipt.anchor_id).expect("read file");

        // The file should contain the root hash and metadata.
        assert!(content.contains(&hex_encode(&root.root_hash)));
        assert!(content.contains("\"leaf_count\": 5"));
        assert!(content.contains("\"created_at\": 3000"));
        assert!(content.contains("\"sequence_number\": 2"));
    }

    // ---- 13. MultiAnchor default ----

    #[test]
    fn multi_anchor_default_is_empty() {
        let multi = MultiAnchor::default();
        let root = make_root(0, 0);
        let results = multi.anchor_all(&root);
        assert!(results.is_empty());
    }

    // ---- 14. AnchorChain find_by_root returns None when empty ----

    #[test]
    fn anchor_chain_find_returns_none_when_empty() {
        let chain = AnchorChain::new();
        assert!(chain.find_by_root(&[0u8; 32]).is_none());
        assert!(chain.find_by_anchor_id("anything").is_none());
        assert!(chain.latest().is_none());
    }
}
