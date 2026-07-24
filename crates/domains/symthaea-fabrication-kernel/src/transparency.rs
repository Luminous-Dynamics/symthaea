// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only Merkle transparency log for release and authority evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const TRANSPARENCY_LOG_SCHEMA: &str = "symthaea.fabrication.transparency-log.v1";
pub const MAX_TRANSPARENCY_ENTRIES: usize = 1_000_000;
pub const MAX_TRANSPARENCY_KIND_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyEntry {
    pub sequence: u64,
    pub recorded_at_unix_s: u64,
    pub kind: String,
    pub subject_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyLog {
    pub schema_version: String,
    pub entries: Vec<TransparencyEntry>,
}

impl Default for TransparencyLog {
    fn default() -> Self {
        Self {
            schema_version: TRANSPARENCY_LOG_SCHEMA.into(),
            entries: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofSide {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InclusionProofNode {
    pub side: ProofSide,
    pub digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyInclusionProof {
    pub tree_size: u64,
    pub leaf_index: u64,
    pub leaf_digest: Sha256Digest,
    pub root_digest: Sha256Digest,
    pub path: Vec<InclusionProofNode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidSequence { expected: u64, actual: u64 },
    TimeRegressed { previous: u64, current: u64 },
    InvalidKind,
    IndexOutOfBounds,
    ProofTooDeep,
    InvalidProof,
    PrefixMismatch,
    Encoding(String),
}

impl TransparencyLog {
    pub fn validate(&self) -> Result<(), TransparencyError> {
        if self.schema_version != TRANSPARENCY_LOG_SCHEMA {
            return Err(TransparencyError::UnsupportedSchema);
        }
        if self.entries.len() > MAX_TRANSPARENCY_ENTRIES {
            return Err(TransparencyError::CapacityExceeded);
        }
        let mut previous_time = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let expected = index as u64 + 1;
            if entry.sequence != expected {
                return Err(TransparencyError::InvalidSequence {
                    expected,
                    actual: entry.sequence,
                });
            }
            validate_kind(&entry.kind)?;
            if previous_time.is_some_and(|time| entry.recorded_at_unix_s < time) {
                return Err(TransparencyError::TimeRegressed {
                    previous: previous_time.unwrap_or_default(),
                    current: entry.recorded_at_unix_s,
                });
            }
            previous_time = Some(entry.recorded_at_unix_s);
        }
        Ok(())
    }

    pub fn append(
        &mut self,
        recorded_at_unix_s: u64,
        kind: impl Into<String>,
        subject_digest: Sha256Digest,
    ) -> Result<Sha256Digest, TransparencyError> {
        self.validate()?;
        if self.entries.len() >= MAX_TRANSPARENCY_ENTRIES {
            return Err(TransparencyError::CapacityExceeded);
        }
        if self
            .entries
            .last()
            .is_some_and(|entry| recorded_at_unix_s < entry.recorded_at_unix_s)
        {
            return Err(TransparencyError::TimeRegressed {
                previous: self
                    .entries
                    .last()
                    .map_or(0, |entry| entry.recorded_at_unix_s),
                current: recorded_at_unix_s,
            });
        }
        let entry = TransparencyEntry {
            sequence: self.entries.len() as u64 + 1,
            recorded_at_unix_s,
            kind: kind.into(),
            subject_digest,
        };
        validate_kind(&entry.kind)?;
        let digest = digest_transparency_entry(&entry)?;
        self.entries.push(entry);
        Ok(digest)
    }

    pub fn root(&self) -> Result<Sha256Digest, TransparencyError> {
        self.validate()?;
        let leaves = self
            .entries
            .iter()
            .map(digest_transparency_entry)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(merkle_root(&leaves))
    }

    pub fn inclusion_proof(
        &self,
        index: usize,
    ) -> Result<TransparencyInclusionProof, TransparencyError> {
        self.validate()?;
        if index >= self.entries.len() {
            return Err(TransparencyError::IndexOutOfBounds);
        }
        let mut level = self
            .entries
            .iter()
            .map(digest_transparency_entry)
            .collect::<Result<Vec<_>, _>>()?;
        let leaf_digest = level[index];
        let mut position = index;
        let mut path = Vec::new();
        while level.len() > 1 {
            if position % 2 == 0 {
                if position + 1 < level.len() {
                    path.push(InclusionProofNode {
                        side: ProofSide::Right,
                        digest: level[position + 1],
                    });
                }
            } else {
                path.push(InclusionProofNode {
                    side: ProofSide::Left,
                    digest: level[position - 1],
                });
            }
            level = next_level(&level);
            position /= 2;
        }
        if path.len() > 64 {
            return Err(TransparencyError::ProofTooDeep);
        }
        Ok(TransparencyInclusionProof {
            tree_size: self.entries.len() as u64,
            leaf_index: index as u64,
            leaf_digest,
            root_digest: level[0],
            path,
        })
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), TransparencyError> {
        previous.validate()?;
        self.validate()?;
        if self.entries.len() < previous.entries.len()
            || self.entries[..previous.entries.len()] != previous.entries
        {
            return Err(TransparencyError::PrefixMismatch);
        }
        Ok(())
    }
}

pub fn digest_transparency_entry(
    entry: &TransparencyEntry,
) -> Result<Sha256Digest, TransparencyError> {
    validate_kind(&entry.kind)?;
    if entry.sequence == 0 {
        return Err(TransparencyError::InvalidSequence {
            expected: 1,
            actual: 0,
        });
    }
    let bytes = serde_json::to_vec(entry)
        .map_err(|error| TransparencyError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-leaf.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_transparency_inclusion(
    proof: &TransparencyInclusionProof,
) -> Result<(), TransparencyError> {
    if proof.tree_size == 0 || proof.leaf_index >= proof.tree_size || proof.path.len() > 64 {
        return Err(TransparencyError::InvalidProof);
    }
    let mut digest = proof.leaf_digest;
    for node in &proof.path {
        digest = match node.side {
            ProofSide::Left => digest_merkle_node(node.digest, digest),
            ProofSide::Right => digest_merkle_node(digest, node.digest),
        };
    }
    if digest != proof.root_digest {
        return Err(TransparencyError::InvalidProof);
    }
    Ok(())
}

pub fn digest_transparency_log(log: &TransparencyLog) -> Result<Sha256Digest, TransparencyError> {
    log.validate()?;
    let root = log.root()?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-log-digest.v1\0");
    hasher.update(&(log.entries.len() as u64).to_le_bytes());
    hasher.update(&root.0);
    Ok(hasher.finalize())
}

fn merkle_root(leaves: &[Sha256Digest]) -> Sha256Digest {
    if leaves.is_empty() {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.transparency-empty.v1\0");
        return hasher.finalize();
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = next_level(&level);
    }
    level[0]
}

fn next_level(level: &[Sha256Digest]) -> Vec<Sha256Digest> {
    let mut output = Vec::with_capacity((level.len() + 1) / 2);
    for pair in level.chunks(2) {
        output.push(if pair.len() == 2 {
            digest_merkle_node(pair[0], pair[1])
        } else {
            pair[0]
        });
    }
    output
}

fn digest_merkle_node(left: Sha256Digest, right: Sha256Digest) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-node.v1\0");
    hasher.update(&left.0);
    hasher.update(&right.0);
    hasher.finalize()
}

fn validate_kind(value: &str) -> Result<(), TransparencyError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_TRANSPARENCY_KIND_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(TransparencyError::InvalidKind);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn every_entry_has_a_valid_inclusion_proof() {
        let mut log = TransparencyLog::default();
        for index in 0..7 {
            log.append(100 + index, "release-candidate", sha256(&[index as u8]))
                .unwrap();
        }
        for index in 0..log.entries.len() {
            verify_transparency_inclusion(&log.inclusion_proof(index).unwrap()).unwrap();
        }
    }

    #[test]
    fn changed_prefix_is_rejected() {
        let mut first = TransparencyLog::default();
        first.append(100, "release", sha256(b"a")).unwrap();
        let mut second = first.clone();
        second.entries[0].subject_digest = sha256(b"b");
        assert_eq!(
            second.verify_successor_of(&first),
            Err(TransparencyError::PrefixMismatch)
        );
    }
}
