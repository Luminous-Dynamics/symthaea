// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! RFC 9162 Merkle Tree Hash proof verification for MACE evaluator witnesses.
//!
//! This module implements only the cryptographic tree layer: domain-separated
//! leaf/node hashing, inclusion proofs and append-only consistency proofs.
//! It does not authenticate a log operator, checkpoint, quorum, timestamp,
//! trust roster, or organizational independence. Those facts must be verified
//! by a separately pinned witness/checkpoint layer.
//!
//! Proof nodes use RFC 9162 ordering: leaf-to-root for inclusion, and the RFC
//! SUBPROOF ordering for consistency. Leaves use SHA-256(0x00 || data) and
//! internal nodes use SHA-256(0x01 || left || right).

use sha2::{Digest, Sha256};

pub const RFC9162_DIGEST_BYTES_V1: usize = 32;
pub const RFC9162_MAX_PROOF_NODES_V1: usize = 64;
pub type Rfc9162DigestV1 = [u8; RFC9162_DIGEST_BYTES_V1];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rfc9162ProofRejectV1 {
    EmptyTree,
    LeafIndexOutOfRange,
    TreeSizeRegression,
    ProofTooLong,
    ProofLengthMismatch,
    InvalidHexDigest,
    InclusionRootMismatch,
    ConsistencyRootMismatch,
    NonEmptyGenesisRoot,
    NonEmptySameSizeProof,
}

/// RFC 9162 leaf hash: SHA-256(0x00 || leaf_data).
pub fn rfc9162_leaf_hash_v1(leaf_data: &[u8]) -> Rfc9162DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update([0x00]);
    hasher.update(leaf_data);
    hasher.finalize().into()
}

/// RFC 9162 internal-node hash: SHA-256(0x01 || left || right).
pub fn rfc9162_node_hash_v1(
    left: &Rfc9162DigestV1,
    right: &Rfc9162DigestV1,
) -> Rfc9162DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update([0x01]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

/// RFC 9162 Merkle Tree Hash for an empty tree.
pub fn rfc9162_empty_tree_hash_v1() -> Rfc9162DigestV1 {
    Sha256::digest([]).into()
}

/// Parse exactly 64 lowercase hexadecimal characters into a SHA-256 digest.
pub fn parse_rfc9162_digest_hex_v1(
    value: &str,
) -> Result<Rfc9162DigestV1, Rfc9162ProofRejectV1> {
    if value.len() != 64 {
        return Err(Rfc9162ProofRejectV1::InvalidHexDigest);
    }
    let mut output = [0u8; RFC9162_DIGEST_BYTES_V1];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or(Rfc9162ProofRejectV1::InvalidHexDigest)?;
        let low = hex_nibble(pair[1]).ok_or(Rfc9162ProofRejectV1::InvalidHexDigest)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

/// Encode a digest as lowercase hexadecimal.
pub fn rfc9162_digest_hex_v1(value: &Rfc9162DigestV1) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

/// Verify one inclusion proof against an already authenticated checkpoint root.
///
/// leaf_hash must already be SHA-256(0x00 || leaf_data). Proof nodes are in
/// RFC 9162 audit-path order, from the leaf toward the root. This proves
/// membership under the supplied root only; it does not authenticate that
/// root or establish that the checkpoint is current.
pub fn verify_rfc9162_inclusion_v1(
    leaf_hash: &Rfc9162DigestV1,
    leaf_index: u64,
    tree_size: u64,
    proof: &[Rfc9162DigestV1],
    expected_root: &Rfc9162DigestV1,
) -> Result<(), Rfc9162ProofRejectV1> {
    if tree_size == 0 {
        return Err(Rfc9162ProofRejectV1::EmptyTree);
    }
    if leaf_index >= tree_size {
        return Err(Rfc9162ProofRejectV1::LeafIndexOutOfRange);
    }
    if proof.len() > RFC9162_MAX_PROOF_NODES_V1 {
        return Err(Rfc9162ProofRejectV1::ProofTooLong);
    }
    let expected_nodes = inclusion_proof_node_count(tree_size, leaf_index);
    if proof.len() != expected_nodes {
        return Err(Rfc9162ProofRejectV1::ProofLengthMismatch);
    }

    let actual_root = inclusion_root_for_subtree(tree_size, leaf_index, leaf_hash, proof)
        .ok_or(Rfc9162ProofRejectV1::ProofLengthMismatch)?;
    if &actual_root != expected_root {
        return Err(Rfc9162ProofRejectV1::InclusionRootMismatch);
    }
    Ok(())
}

fn inclusion_root_for_subtree(
    tree_size: u64,
    leaf_index: u64,
    leaf_hash: &Rfc9162DigestV1,
    proof: &[Rfc9162DigestV1],
) -> Option<Rfc9162DigestV1> {
    if tree_size == 1 {
        return if leaf_index == 0 && proof.is_empty() {
            Some(*leaf_hash)
        } else {
            None
        };
    }

    let split = largest_power_of_two_less_than(tree_size)?;
    if leaf_index < split {
        let inner_nodes = inclusion_proof_node_count(split, leaf_index);
        if proof.len() != inner_nodes.checked_add(1)? {
            return None;
        }
        let left = inclusion_root_for_subtree(
            split,
            leaf_index,
            leaf_hash,
            &proof[..inner_nodes],
        )?;
        Some(rfc9162_node_hash_v1(&left, &proof[inner_nodes]))
    } else {
        let right_size = tree_size.checked_sub(split)?;
        let right_index = leaf_index.checked_sub(split)?;
        let inner_nodes = inclusion_proof_node_count(right_size, right_index);
        if proof.len() != inner_nodes.checked_add(1)? {
            return None;
        }
        let right = inclusion_root_for_subtree(
            right_size,
            right_index,
            leaf_hash,
            &proof[..inner_nodes],
        )?;
        Some(rfc9162_node_hash_v1(&proof[inner_nodes], &right))
    }
}

fn inclusion_proof_node_count(tree_size: u64, leaf_index: u64) -> usize {
    if tree_size <= 1 {
        return 0;
    }
    let Some(split) = largest_power_of_two_less_than(tree_size) else {
        return 0;
    };
    let inner = if leaf_index < split {
        inclusion_proof_node_count(split, leaf_index)
    } else {
        inclusion_proof_node_count(tree_size - split, leaf_index - split)
    };
    inner.saturating_add(1)
}

fn largest_power_of_two_less_than(value: u64) -> Option<u64> {
    if value <= 1 {
        return None;
    }
    let mut power = 1u64;
    while power <= (value - 1) / 2 {
        power = power.checked_mul(2)?;
    }
    Some(power)
}

/// Verify an RFC 9162 consistency proof from a pinned old checkpoint to a
/// candidate new checkpoint. Smaller new trees are always rejected.
pub fn verify_rfc9162_consistency_v1(
    old_size: u64,
    new_size: u64,
    old_root: &Rfc9162DigestV1,
    new_root: &Rfc9162DigestV1,
    proof: &[Rfc9162DigestV1],
) -> Result<(), Rfc9162ProofRejectV1> {
    if proof.len() > RFC9162_MAX_PROOF_NODES_V1 {
        return Err(Rfc9162ProofRejectV1::ProofTooLong);
    }
    if old_size > new_size {
        return Err(Rfc9162ProofRejectV1::TreeSizeRegression);
    }
    if old_size == 0 {
        if old_root != &rfc9162_empty_tree_hash_v1() {
            return Err(Rfc9162ProofRejectV1::NonEmptyGenesisRoot);
        }
        if !proof.is_empty() {
            return Err(Rfc9162ProofRejectV1::ProofLengthMismatch);
        }
        return Ok(());
    }
    if old_size == new_size {
        if !proof.is_empty() {
            return Err(Rfc9162ProofRejectV1::NonEmptySameSizeProof);
        }
        return if old_root == new_root {
            Ok(())
        } else {
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        };
    }
    if proof.is_empty() {
        return Err(Rfc9162ProofRejectV1::ProofLengthMismatch);
    }

    let mut first_node = old_size - 1;
    let mut second_node = new_size - 1;
    while first_node & 1 == 1 {
        first_node >>= 1;
        second_node >>= 1;
    }

    let (mut first_root, mut second_root, proof_start) = if first_node == 0 {
        (*old_root, *old_root, 0usize)
    } else {
        (proof[0], proof[0], 1usize)
    };

    for proof_node in &proof[proof_start..] {
        if second_node == 0 {
            return Err(Rfc9162ProofRejectV1::ProofLengthMismatch);
        }
        if (first_node & 1) == 1 || first_node == second_node {
            first_root = rfc9162_node_hash_v1(proof_node, &first_root);
            second_root = rfc9162_node_hash_v1(proof_node, &second_root);
            while (first_node & 1) == 0 && first_node != 0 {
                first_node >>= 1;
                second_node >>= 1;
            }
        } else {
            second_root = rfc9162_node_hash_v1(&second_root, proof_node);
        }
        first_node >>= 1;
        second_node >>= 1;
    }

    if second_node == 0 && first_root == *old_root && second_root == *new_root {
        Ok(())
    } else {
        Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_leaves(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|index| format!("mace-rfc9162-fixture-leaf-{index}").into_bytes())
            .collect()
    }

    fn tree_hash(leaves: &[Vec<u8>]) -> Rfc9162DigestV1 {
        match leaves.len() {
            0 => rfc9162_empty_tree_hash_v1(),
            1 => rfc9162_leaf_hash_v1(&leaves[0]),
            count => {
                let split = largest_power_of_two_less_than(count as u64)
                    .expect("non-singleton tree has split");
                let split = usize::try_from(split).expect("split fits usize");
                let left = tree_hash(&leaves[..split]);
                let right = tree_hash(&leaves[split..]);
                rfc9162_node_hash_v1(&left, &right)
            }
        }
    }

    fn inclusion_proof(leaves: &[Vec<u8>], leaf_index: usize) -> Vec<Rfc9162DigestV1> {
        assert!(leaf_index < leaves.len());
        if leaves.len() == 1 {
            return Vec::new();
        }
        let split = usize::try_from(
            largest_power_of_two_less_than(leaves.len() as u64).expect("split"),
        )
        .expect("split fits usize");
        if leaf_index < split {
            let mut proof = inclusion_proof(&leaves[..split], leaf_index);
            proof.push(tree_hash(&leaves[split..]));
            proof
        } else {
            let mut proof = inclusion_proof(&leaves[split..], leaf_index - split);
            proof.push(tree_hash(&leaves[..split]));
            proof
        }
    }

    /// RFC 9162 SUBPROOF(m, D[n], b), used only by tests to create proofs.
    fn consistency_subproof(
        old_size: usize,
        leaves: &[Vec<u8>],
        complete_subtree: bool,
    ) -> Vec<Rfc9162DigestV1> {
        if old_size == leaves.len() {
            return if complete_subtree {
                Vec::new()
            } else {
                vec![tree_hash(leaves)]
            };
        }
        let split = usize::try_from(
            largest_power_of_two_less_than(leaves.len() as u64).expect("split"),
        )
        .expect("split fits usize");
        if old_size <= split {
            let mut proof =
                consistency_subproof(old_size, &leaves[..split], complete_subtree);
            proof.push(tree_hash(&leaves[split..]));
            proof
        } else {
            let mut proof =
                consistency_subproof(old_size - split, &leaves[split..], false);
            proof.push(tree_hash(&leaves[..split]));
            proof
        }
    }

    #[test]
    fn inclusion_proofs_verify_for_every_leaf_in_tree_sizes_one_through_sixty_four() {
        let mut verified = 0usize;
        for size in 1..=64usize {
            let leaves = fixture_leaves(size);
            let root = tree_hash(&leaves);
            for index in 0..size {
                let proof = inclusion_proof(&leaves, index);
                let leaf = rfc9162_leaf_hash_v1(&leaves[index]);
                verify_rfc9162_inclusion_v1(
                    &leaf,
                    index as u64,
                    size as u64,
                    &proof,
                    &root,
                )
                .expect("valid inclusion proof");
                verified += 1;
            }
        }
        assert_eq!(verified, 2_080);
    }

    #[test]
    fn consistency_proofs_verify_for_every_old_new_size_pair_through_sixty_four() {
        let mut verified = 0usize;
        for new_size in 2..=64usize {
            let leaves = fixture_leaves(new_size);
            let new_root = tree_hash(&leaves);
            for old_size in 1..new_size {
                let old_root = tree_hash(&leaves[..old_size]);
                let proof = consistency_subproof(old_size, &leaves, true);
                verify_rfc9162_consistency_v1(
                    old_size as u64,
                    new_size as u64,
                    &old_root,
                    &new_root,
                    &proof,
                )
                .expect("valid consistency proof");
                verified += 1;
            }
        }
        assert_eq!(verified, 2_016);
    }

    #[test]
    fn fixed_python_reference_vectors_verify_rfc9162_inclusion_and_consistency() {
        let leaves = fixture_leaves(13);
        let new_root = tree_hash(&leaves);
        assert_eq!(
            rfc9162_digest_hex_v1(&new_root),
            "6adfcbd4eb5db2afdffcdaf31ee4231e0b899dc28228716130b5d3f202ceddf8"
        );

        let leaf = rfc9162_leaf_hash_v1(&leaves[7]);
        assert_eq!(
            rfc9162_digest_hex_v1(&leaf),
            "806c793bf78be40e761f9baf94ff64166614a684e190ee363e8687711e345901"
        );
        let inclusion_hex = [
            "0a78e48e15fa16087f065ff1c7f3843d5811893c933b638f573244031a3d7d22",
            "936b56edc35add7c6c00a9ba8462af6de6ff9af1aa45c520d0f505d8cb64074d",
            "a277d6157e3259f4b8fa06c37919cfe187887a6c48c9be47ef6e122673108bd8",
            "acc53ef4dc6e4231066f998755ead88abcb3fbecc7b269fb36c1761b869529ef",
        ];
        let inclusion: Vec<_> = inclusion_hex
            .iter()
            .map(|value| parse_rfc9162_digest_hex_v1(value).unwrap())
            .collect();
        verify_rfc9162_inclusion_v1(&leaf, 7, 13, &inclusion, &new_root)
            .expect("published independent inclusion vector");

        let old_root = tree_hash(&leaves[..11]);
        assert_eq!(
            rfc9162_digest_hex_v1(&old_root),
            "7f199e2a6d6e4a05e029092dd52e806710ab4068200d11b5108360e2f84e3517"
        );
        let consistency_hex = [
            "0668bf3d2da641fb778ed7de825d8c2bac5d0b1852e1b3cffcd88d714378feca",
            "69445a2b0b81ca07ba1669623a1b436d3fa7b77256494bd416d8dbd85eab3b09",
            "d72998b280a4b46d0fb053c9e5dd4f41588fcb9e527b87ce12853c5b181822d0",
            "7ea68c137f8fa8f3c2dc6ebb65b8f17b82a5b56e94496208a57673ac48893ee5",
            "ad8c19924a5c0b3b61e2a9203ee62c01f90227328990e1a16cb37ec2e7b85c8f",
        ];
        let consistency: Vec<_> = consistency_hex
            .iter()
            .map(|value| parse_rfc9162_digest_hex_v1(value).unwrap())
            .collect();
        verify_rfc9162_consistency_v1(11, 13, &old_root, &new_root, &consistency)
            .expect("published independent consistency vector");
    }

    #[test]
    fn valid_genesis_extension_requires_the_exact_empty_root_and_no_proof() {
        let leaves = fixture_leaves(7);
        let new_root = tree_hash(&leaves);
        verify_rfc9162_consistency_v1(
            0,
            leaves.len() as u64,
            &rfc9162_empty_tree_hash_v1(),
            &new_root,
            &[],
        )
        .expect("genesis extension");
    }

    #[test]
    fn same_size_checkpoint_requires_identical_root_and_empty_proof() {
        let leaves = fixture_leaves(9);
        let root = tree_hash(&leaves);
        assert_eq!(
            verify_rfc9162_consistency_v1(9, 9, &root, &root, &[]),
            Ok(())
        );
        let other = tree_hash(&fixture_leaves(8));
        assert_eq!(
            verify_rfc9162_consistency_v1(9, 9, &root, &other, &[]),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_consistency_v1(9, 9, &root, &root, &[root]),
            Err(Rfc9162ProofRejectV1::NonEmptySameSizeProof)
        );
    }

    #[test]
    fn inclusion_rejects_wrong_leaf_root_index_and_truncated_or_extended_paths() {
        let leaves = fixture_leaves(13);
        let root = tree_hash(&leaves);
        let proof = inclusion_proof(&leaves, 7);
        let leaf = rfc9162_leaf_hash_v1(&leaves[7]);
        let wrong_leaf = rfc9162_leaf_hash_v1(b"not-the-committed-leaf");
        let wrong_root = tree_hash(&fixture_leaves(12));

        assert_eq!(
            verify_rfc9162_inclusion_v1(&wrong_leaf, 7, 13, &proof, &root),
            Err(Rfc9162ProofRejectV1::InclusionRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_inclusion_v1(&leaf, 7, 13, &proof, &wrong_root),
            Err(Rfc9162ProofRejectV1::InclusionRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_inclusion_v1(&leaf, 13, 13, &proof, &root),
            Err(Rfc9162ProofRejectV1::LeafIndexOutOfRange)
        );
        assert_eq!(
            verify_rfc9162_inclusion_v1(&leaf, 7, 13, &proof[..proof.len() - 1], &root),
            Err(Rfc9162ProofRejectV1::ProofLengthMismatch)
        );
        let mut extended = proof.clone();
        extended.push(root);
        assert_eq!(
            verify_rfc9162_inclusion_v1(&leaf, 7, 13, &extended, &root),
            Err(Rfc9162ProofRejectV1::ProofLengthMismatch)
        );
        assert_eq!(
            verify_rfc9162_inclusion_v1(&leaf, 0, 0, &[], &root),
            Err(Rfc9162ProofRejectV1::EmptyTree)
        );
    }

    #[test]
    fn consistency_rejects_wrong_roots_truncation_and_extension() {
        let leaves = fixture_leaves(25);
        let old_root = tree_hash(&leaves[..11]);
        let new_root = tree_hash(&leaves);
        let proof = consistency_subproof(11, &leaves, true);
        let wrong_old = tree_hash(&leaves[..10]);
        let wrong_new = tree_hash(&fixture_leaves(24));

        assert_eq!(
            verify_rfc9162_consistency_v1(11, 25, &wrong_old, &new_root, &proof),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_consistency_v1(11, 25, &old_root, &wrong_new, &proof),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_consistency_v1(11, 25, &old_root, &new_root, &proof[..proof.len() - 1]),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
        let mut extended = proof.clone();
        extended.push(old_root);
        assert_eq!(
            verify_rfc9162_consistency_v1(11, 25, &old_root, &new_root, &extended),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
        assert_eq!(
            verify_rfc9162_consistency_v1(12, 11, &old_root, &new_root, &[]),
            Err(Rfc9162ProofRejectV1::TreeSizeRegression)
        );
        assert_eq!(
            verify_rfc9162_consistency_v1(1, 1, &old_root, &wrong_new, &[]),
            Err(Rfc9162ProofRejectV1::ConsistencyRootMismatch)
        );
    }

    #[test]
    fn digest_wire_parser_is_lowercase_and_exact_length() {
        let digest = [0xab; RFC9162_DIGEST_BYTES_V1];
        let encoded = rfc9162_digest_hex_v1(&digest);
        assert_eq!(parse_rfc9162_digest_hex_v1(&encoded), Ok(digest));
        assert_eq!(
            parse_rfc9162_digest_hex_v1(&encoded.to_uppercase()),
            Err(Rfc9162ProofRejectV1::InvalidHexDigest)
        );
        assert_eq!(
            parse_rfc9162_digest_hex_v1(&encoded[..63]),
            Err(Rfc9162ProofRejectV1::InvalidHexDigest)
        );
        assert_eq!(
            parse_rfc9162_digest_hex_v1(&format!("{}g", &encoded[..63])),
            Err(Rfc9162ProofRejectV1::InvalidHexDigest)
        );
    }
}
