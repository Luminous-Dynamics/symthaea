// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic Hashing Utilities
//!
//! Provides BLAKE3 hashing for datasets with streaming support for large files.

use crate::{Error, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Hash algorithm identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// BLAKE3 (default, fast and secure)
    Blake3,
    /// SHA-256 (for compatibility)
    Sha256,
}

impl HashAlgorithm {
    /// Get the string representation of the algorithm
    pub fn as_str(&self) -> &'static str {
        match self {
            HashAlgorithm::Blake3 => "blake3",
            HashAlgorithm::Sha256 => "sha256",
        }
    }
}

/// Hash result with algorithm identifier
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hash {
    /// Algorithm used
    pub algorithm: HashAlgorithm,
    /// Hash bytes
    pub bytes: Vec<u8>,
}

impl Hash {
    /// Create a new hash
    pub fn new(algorithm: HashAlgorithm, bytes: Vec<u8>) -> Self {
        Self { algorithm, bytes }
    }

    /// Convert to hex string with algorithm prefix (e.g., "blake3:abc123...")
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.algorithm.as_str(), hex::encode(&self.bytes))
    }

    /// Parse from prefixed hex string
    pub fn from_string(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(Error::Generic(
                "Invalid hash format, expected 'algorithm:hexdigest'".to_string(),
            ));
        }

        let algorithm = match parts[0] {
            "blake3" => HashAlgorithm::Blake3,
            "sha256" => HashAlgorithm::Sha256,
            _ => {
                return Err(Error::Generic(format!(
                    "Unknown hash algorithm: {}",
                    parts[0]
                )))
            }
        };

        let bytes = hex::decode(parts[1])
            .map_err(|e| Error::Generic(format!("Invalid hex string: {}", e)))?;

        Ok(Self { algorithm, bytes })
    }

    /// Get the hash as a hex string (without algorithm prefix)
    pub fn hex(&self) -> String {
        hex::encode(&self.bytes)
    }
}

/// Hash a file using BLAKE3
pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<Hash> {
    hash_file_with_algorithm(path, HashAlgorithm::Blake3)
}

/// Hash a file with a specific algorithm
pub fn hash_file_with_algorithm<P: AsRef<Path>>(
    path: P,
    algorithm: HashAlgorithm,
) -> Result<Hash> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Generic(format!("Failed to open file: {}", e)))?;

    let mut reader = BufReader::new(file);

    match algorithm {
        HashAlgorithm::Blake3 => {
            let mut hasher = blake3::Hasher::new();
            let mut buffer = vec![0u8; 65536]; // 64KB buffer

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| Error::Generic(format!("Failed to read file: {}", e)))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            let hash = hasher.finalize();
            Ok(Hash::new(algorithm, hash.as_bytes().to_vec()))
        }
        HashAlgorithm::Sha256 => {
            use sha2::{Digest, Sha256};

            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 65536];

            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| Error::Generic(format!("Failed to read file: {}", e)))?;

                if bytes_read == 0 {
                    break;
                }

                hasher.update(&buffer[..bytes_read]);
            }

            let hash = hasher.finalize();
            Ok(Hash::new(algorithm, hash.to_vec()))
        }
    }
}

/// Hash bytes with BLAKE3
pub fn hash_bytes(data: &[u8]) -> Hash {
    hash_bytes_with_algorithm(data, HashAlgorithm::Blake3)
}

/// Hash bytes with a specific algorithm
pub fn hash_bytes_with_algorithm(data: &[u8], algorithm: HashAlgorithm) -> Hash {
    match algorithm {
        HashAlgorithm::Blake3 => {
            let hash = blake3::hash(data);
            Hash::new(algorithm, hash.as_bytes().to_vec())
        }
        HashAlgorithm::Sha256 => {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(data);
            let hash = hasher.finalize();
            Hash::new(algorithm, hash.to_vec())
        }
    }
}

/// Verify a file against a known hash
pub fn verify_file<P: AsRef<Path>>(path: P, expected_hash: &Hash) -> Result<bool> {
    let computed_hash = hash_file_with_algorithm(path, expected_hash.algorithm)?;
    Ok(computed_hash.bytes == expected_hash.bytes)
}

/// Merkle tree node for large dataset verification
#[derive(Debug, Clone)]
pub struct MerkleNode {
    /// Hash of this node
    pub hash: Hash,
    /// Left child (if internal node)
    pub left: Option<Box<MerkleNode>>,
    /// Right child (if internal node)
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Create a leaf node
    pub fn leaf(hash: Hash) -> Self {
        Self {
            hash,
            left: None,
            right: None,
        }
    }

    /// Create an internal node from two children
    pub fn internal(left: MerkleNode, right: MerkleNode) -> Self {
        // Combine hashes
        let combined = [left.hash.bytes.as_slice(), right.hash.bytes.as_slice()].concat();
        let hash = hash_bytes_with_algorithm(&combined, left.hash.algorithm);

        Self {
            hash,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    /// Get the root hash
    pub fn root_hash(&self) -> &Hash {
        &self.hash
    }
}

/// Build a Merkle tree from a list of hashes
pub fn build_merkle_tree(mut hashes: Vec<Hash>) -> Result<MerkleNode> {
    if hashes.is_empty() {
        return Err(Error::Generic("Cannot build Merkle tree from empty list".to_string()));
    }

    // Create leaf nodes
    let mut nodes: Vec<MerkleNode> = hashes.drain(..).map(MerkleNode::leaf).collect();

    // Build tree bottom-up
    while nodes.len() > 1 {
        let mut next_level = Vec::new();

        for chunk in nodes.chunks(2) {
            if chunk.len() == 2 {
                next_level.push(MerkleNode::internal(chunk[0].clone(), chunk[1].clone()));
            } else {
                // Odd number of nodes, promote the last one
                next_level.push(chunk[0].clone());
            }
        }

        nodes = next_level;
    }

    // Safe: we checked for empty list at start, and loop maintains at least one node
    nodes.into_iter().next().ok_or_else(|| Error::Generic("Merkle tree construction failed unexpectedly".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_hash_bytes() {
        let data = b"Hello, Mycelix-DeSci!";
        let hash = hash_bytes(data);

        assert_eq!(hash.algorithm, HashAlgorithm::Blake3);
        assert!(!hash.bytes.is_empty());
    }

    #[test]
    fn test_hash_string_roundtrip() {
        let data = b"Test data";
        let hash = hash_bytes(data);
        let hash_str = hash.to_string();

        assert!(hash_str.starts_with("blake3:"));

        let parsed = Hash::from_string(&hash_str).unwrap();
        assert_eq!(parsed.algorithm, hash.algorithm);
        assert_eq!(parsed.bytes, hash.bytes);
    }

    #[test]
    fn test_hash_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Test file content").unwrap();

        let hash = hash_file(temp_file.path()).unwrap();
        assert_eq!(hash.algorithm, HashAlgorithm::Blake3);

        // Verify hash
        let is_valid = verify_file(temp_file.path(), &hash).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_different_algorithms() {
        let data = b"Same data, different algorithms";

        let blake3_hash = hash_bytes_with_algorithm(data, HashAlgorithm::Blake3);
        let sha256_hash = hash_bytes_with_algorithm(data, HashAlgorithm::Sha256);

        assert_ne!(blake3_hash.bytes, sha256_hash.bytes);
        assert_eq!(blake3_hash.algorithm, HashAlgorithm::Blake3);
        assert_eq!(sha256_hash.algorithm, HashAlgorithm::Sha256);
    }

    #[test]
    fn test_merkle_tree() {
        let hashes = vec![
            hash_bytes(b"block1"),
            hash_bytes(b"block2"),
            hash_bytes(b"block3"),
            hash_bytes(b"block4"),
        ];

        let tree = build_merkle_tree(hashes).unwrap();
        let root_hash = tree.root_hash();

        assert!(!root_hash.bytes.is_empty());
    }

    #[test]
    fn test_merkle_tree_single_hash() {
        let hashes = vec![hash_bytes(b"single block")];
        let tree = build_merkle_tree(hashes).unwrap();

        assert!(tree.left.is_none());
        assert!(tree.right.is_none());
    }
}
