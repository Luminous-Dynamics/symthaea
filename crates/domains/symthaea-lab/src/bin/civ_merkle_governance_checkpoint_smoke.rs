//! SYM-CIV-008: Merkle governance checkpoint and proof smoke.
//!
//! Implements the SHA-256 Merkle Tree Hash shape specified by RFC 9162:
//! leaves use HASH(0x00 || leaf), internal nodes use HASH(0x01 || left || right),
//! and the tree shape is determined by the number of entries.
//!
//! This executable tests local root, inclusion-proof, and consistency-proof
//! algorithms. It does not implement signed tree heads, witness gossip,
//! network transport, or a production transparency service.

use sha2::{Digest as ShaDigest, Sha256};

type Hash = [u8; 32];

#[derive(Clone, Debug, Eq, PartialEq)]
struct TreeHead {
    log_id: String,
    tree_size: usize,
    root_hash: Hash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProofFailure {
    EmptyTree,
    InvalidLeafIndex,
    InvalidTreeSize,
    WrongRoot,
    InvalidProof,
    ForkDetected,
    DifferentLog,
    ConsistencyProofRequired,
}

fn leaf_hash(entry: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x00_u8]);
    hasher.update(entry);
    hasher.finalize().into()
}

fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update([0x01_u8]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn largest_power_of_two_less_than(n: usize) -> usize {
    debug_assert!(n > 1);
    let mut power = 1_usize;
    while power * 2 < n {
        power *= 2;
    }
    power
}

fn merkle_tree_hash(entries: &[Vec<u8>]) -> Hash {
    match entries.len() {
        0 => Sha256::digest([]).into(),
        1 => leaf_hash(&entries[0]),
        n => {
            let k = largest_power_of_two_less_than(n);
            node_hash(
                &merkle_tree_hash(&entries[..k]),
                &merkle_tree_hash(&entries[k..]),
            )
        }
    }
}

fn inclusion_path(index: usize, entries: &[Vec<u8>]) -> Option<Vec<Hash>> {
    if index >= entries.len() {
        return None;
    }
    Some(inclusion_path_inner(index, entries))
}

fn inclusion_path_inner(index: usize, entries: &[Vec<u8>]) -> Vec<Hash> {
    if entries.len() == 1 {
        return Vec::new();
    }

    let k = largest_power_of_two_less_than(entries.len());
    if index < k {
        let mut proof = inclusion_path_inner(index, &entries[..k]);
        proof.push(merkle_tree_hash(&entries[k..]));
        proof
    } else {
        let mut proof = inclusion_path_inner(index - k, &entries[k..]);
        proof.push(merkle_tree_hash(&entries[..k]));
        proof
    }
}

fn verify_inclusion(
    entry: &[u8],
    leaf_index: usize,
    tree_size: usize,
    inclusion_proof: &[Hash],
    expected_root: &Hash,
) -> Result<(), ProofFailure> {
    if tree_size == 0 {
        return Err(ProofFailure::EmptyTree);
    }
    if leaf_index >= tree_size {
        return Err(ProofFailure::InvalidLeafIndex);
    }

    let mut fn_index = leaf_index;
    let mut sn_index = tree_size - 1;
    let mut root = leaf_hash(entry);

    for proof_node in inclusion_proof {
        if sn_index == 0 {
            return Err(ProofFailure::InvalidProof);
        }

        if (fn_index & 1) == 1 || fn_index == sn_index {
            root = node_hash(proof_node, &root);
            if (fn_index & 1) == 0 {
                while fn_index != 0 && (fn_index & 1) == 0 {
                    fn_index >>= 1;
                    sn_index >>= 1;
                }
            }
        } else {
            root = node_hash(&root, proof_node);
        }

        fn_index >>= 1;
        sn_index >>= 1;
    }

    if sn_index != 0 {
        return Err(ProofFailure::InvalidProof);
    }
    if &root != expected_root {
        return Err(ProofFailure::WrongRoot);
    }
    Ok(())
}

fn consistency_subproof(
    old_size: usize,
    entries: &[Vec<u8>],
    complete_subtree: bool,
) -> Vec<Hash> {
    let n = entries.len();
    if old_size == n {
        return if complete_subtree {
            Vec::new()
        } else {
            vec![merkle_tree_hash(entries)]
        };
    }

    let k = largest_power_of_two_less_than(n);
    if old_size <= k {
        let mut proof = consistency_subproof(old_size, &entries[..k], complete_subtree);
        proof.push(merkle_tree_hash(&entries[k..]));
        proof
    } else {
        let mut proof = consistency_subproof(old_size - k, &entries[k..], false);
        proof.push(merkle_tree_hash(&entries[..k]));
        proof
    }
}

fn generate_consistency_proof(
    old_size: usize,
    entries: &[Vec<u8>],
) -> Result<Vec<Hash>, ProofFailure> {
    if old_size > entries.len() {
        return Err(ProofFailure::InvalidTreeSize);
    }
    if old_size == 0 || old_size == entries.len() {
        return Ok(Vec::new());
    }
    Ok(consistency_subproof(old_size, entries, true))
}

fn verify_consistency(
    old_size: usize,
    new_size: usize,
    old_root: &Hash,
    new_root: &Hash,
    consistency_proof: &[Hash],
) -> Result<(), ProofFailure> {
    if old_size > new_size {
        return Err(ProofFailure::InvalidTreeSize);
    }

    let empty_root: Hash = Sha256::digest([]).into();
    if old_size == 0 {
        if old_root != &empty_root || !consistency_proof.is_empty() {
            return Err(ProofFailure::InvalidProof);
        }
        return Ok(());
    }

    if old_size == new_size {
        if !consistency_proof.is_empty() {
            return Err(ProofFailure::InvalidProof);
        }
        return if old_root == new_root {
            Ok(())
        } else {
            Err(ProofFailure::WrongRoot)
        };
    }

    if new_size == 0 || consistency_proof.is_empty() {
        return Err(ProofFailure::InvalidProof);
    }

    let mut proof_nodes = Vec::with_capacity(consistency_proof.len() + 1);
    if old_size.is_power_of_two() {
        proof_nodes.push(*old_root);
    }
    proof_nodes.extend_from_slice(consistency_proof);

    let mut fn_index = old_size - 1;
    let mut sn_index = new_size - 1;
    while (fn_index & 1) == 1 {
        fn_index >>= 1;
        sn_index >>= 1;
    }

    let Some(first_node) = proof_nodes.first() else {
        return Err(ProofFailure::InvalidProof);
    };
    let mut first_root = *first_node;
    let mut second_root = *first_node;

    for proof_node in proof_nodes.iter().skip(1) {
        if sn_index == 0 {
            return Err(ProofFailure::InvalidProof);
        }

        if (fn_index & 1) == 1 || fn_index == sn_index {
            first_root = node_hash(proof_node, &first_root);
            second_root = node_hash(proof_node, &second_root);

            if (fn_index & 1) == 0 {
                while fn_index != 0 && (fn_index & 1) == 0 {
                    fn_index >>= 1;
                    sn_index >>= 1;
                }
            }
        } else {
            second_root = node_hash(&second_root, proof_node);
        }

        fn_index >>= 1;
        sn_index >>= 1;
    }

    if sn_index != 0 || first_root != *old_root || second_root != *new_root {
        return Err(ProofFailure::WrongRoot);
    }
    Ok(())
}

fn make_tree_head(log_id: &str, entries: &[Vec<u8>]) -> TreeHead {
    TreeHead {
        log_id: log_id.to_owned(),
        tree_size: entries.len(),
        root_hash: merkle_tree_hash(entries),
    }
}

fn compare_tree_heads(first: &TreeHead, second: &TreeHead) -> Result<(), ProofFailure> {
    if first.log_id != second.log_id {
        return Err(ProofFailure::DifferentLog);
    }
    if first.tree_size == second.tree_size {
        return if first.root_hash == second.root_hash {
            Ok(())
        } else {
            Err(ProofFailure::ForkDetected)
        };
    }
    Err(ProofFailure::ConsistencyProofRequired)
}

fn encode_field(output: &mut Vec<u8>, field: &[u8]) {
    output.extend_from_slice(&(field.len() as u64).to_be_bytes());
    output.extend_from_slice(field);
}

fn canonical_governance_event(
    log_id: &str,
    sequence: u64,
    actor: &str,
    action: &str,
    target: &str,
    scope: &str,
    reason: &str,
    policy_version: &str,
    effective_epoch: u64,
) -> Vec<u8> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(b"mycelix-civ-governance-merkle-entry-v1\0");
    encode_field(&mut encoded, log_id.as_bytes());
    encoded.extend_from_slice(&sequence.to_be_bytes());
    encode_field(&mut encoded, actor.as_bytes());
    encode_field(&mut encoded, action.as_bytes());
    encode_field(&mut encoded, target.as_bytes());
    encode_field(&mut encoded, scope.as_bytes());
    encode_field(&mut encoded, reason.as_bytes());
    encode_field(&mut encoded, policy_version.as_bytes());
    encoded.extend_from_slice(&effective_epoch.to_be_bytes());
    encoded
}

fn sample_entries(log_id: &str, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|sequence| {
            canonical_governance_event(
                log_id,
                sequence as u64 + 1,
                "authority-civ-1",
                if sequence % 2 == 0 { "authorize" } else { "suspend" },
                "deployment-v1",
                "critical-scope-v1",
                if sequence % 2 == 0 { "approved" } else { "safeguard" },
                "policy-v1",
                100 + sequence as u64,
            )
        })
        .collect()
}

fn main() {
    // Merkle domain separation and edge-case roots.
    let empty: Vec<Vec<u8>> = Vec::new();
    assert_eq!(merkle_tree_hash(&empty), Sha256::digest([]).into());
    assert_ne!(leaf_hash(b"event-a"), node_hash(&leaf_hash(b"event-a"), &leaf_hash(b"event-b")));
    assert_ne!(leaf_hash(b"event-a"), leaf_hash(b"event-b"));

    // Validate every inclusion proof over power-of-two and irregular tree sizes.
    for tree_size in 1..=17 {
        let entries = sample_entries("civ-log-v1", tree_size);
        let root = merkle_tree_hash(&entries);

        for index in 0..tree_size {
            let proof = inclusion_path(index, &entries).expect("valid leaf index");
            assert_eq!(
                verify_inclusion(&entries[index], index, tree_size, &proof, &root),
                Ok(()),
                "inclusion failed for size {tree_size}, index {index}"
            );

            let mut changed_leaf = entries[index].clone();
            changed_leaf.push(0xFF);
            assert_eq!(
                verify_inclusion(&changed_leaf, index, tree_size, &proof, &root),
                Err(ProofFailure::WrongRoot)
            );
        }

        assert_eq!(
            verify_inclusion(
                b"no such leaf",
                tree_size,
                tree_size,
                &[],
                &root
            ),
            Err(ProofFailure::InvalidLeafIndex)
        );

        if tree_size > 1 {
            let mut proof = inclusion_path(0, &entries).expect("first leaf proof");
            proof[0][0] ^= 0x01;
            assert_eq!(
                verify_inclusion(&entries[0], 0, tree_size, &proof, &root),
                Err(ProofFailure::WrongRoot)
            );
        }

        let mut extra_proof = inclusion_path(tree_size - 1, &entries)
            .expect("last leaf proof");
        extra_proof.push([0x11; 32]);
        assert_eq!(
            verify_inclusion(
                &entries[tree_size - 1],
                tree_size - 1,
                tree_size,
                &extra_proof,
                &root
            ),
            Err(ProofFailure::InvalidProof)
        );
    }

    // Validate every prior-size consistency proof for each larger tree.
    for new_size in 1..=17 {
        let entries = sample_entries("civ-log-v1", new_size);
        let new_root = merkle_tree_hash(&entries);

        assert_eq!(
            verify_consistency(0, new_size, &merkle_tree_hash(&empty), &new_root, &[]),
            Ok(())
        );

        for old_size in 1..=new_size {
            let old_root = merkle_tree_hash(&entries[..old_size]);
            let proof = generate_consistency_proof(old_size, &entries)
                .expect("old size must not exceed new size");
            assert_eq!(
                verify_consistency(old_size, new_size, &old_root, &new_root, &proof),
                Ok(()),
                "consistency failed from {old_size} to {new_size}"
            );

            if old_size < new_size && !proof.is_empty() {
                let mut tampered_proof = proof.clone();
                tampered_proof[0][0] ^= 0x80;
                assert!(
                    verify_consistency(old_size, new_size, &old_root, &new_root, &tampered_proof)
                        .is_err(),
                    "tampered proof accepted from {old_size} to {new_size}"
                );
            }
        }

        let same_root = merkle_tree_hash(&entries);
        assert_eq!(
            verify_consistency(
                new_size,
                new_size,
                &same_root,
                &same_root,
                &[]
            ),
            Ok(())
        );
    }

    // Prefix mutation must break consistency against the original checkpoint.
    let original = sample_entries("civ-log-v1", 9);
    let old_size = 4;
    let old_root = merkle_tree_hash(&original[..old_size]);
    let new_root = merkle_tree_hash(&original);
    let proof = generate_consistency_proof(old_size, &original).expect("valid prefix");
    assert_eq!(
        verify_consistency(old_size, original.len(), &old_root, &new_root, &proof),
        Ok(())
    );

    let mut rewritten_prefix = original.clone();
    rewritten_prefix[1].push(0x42);
    let rewritten_root = merkle_tree_hash(&rewritten_prefix);
    assert_ne!(rewritten_root, new_root);
    assert!(
        verify_consistency(old_size, rewritten_prefix.len(), &old_root, &rewritten_root, &proof)
            .is_err()
    );

    // A same-size fork is detected by comparing advertised roots; a proof can
    // establish prefix consistency but cannot determine which conflicting head
    // an operator showed to another client.
    let log_a_entries = sample_entries("civ-log-v1", 7);
    let mut log_b_entries = log_a_entries.clone();
    log_b_entries[6].push(0x99);
    let head_a = make_tree_head("civ-log-v1", &log_a_entries);
    let head_b = make_tree_head("civ-log-v1", &log_b_entries);
    assert_eq!(
        compare_tree_heads(&head_a, &head_b),
        Err(ProofFailure::ForkDetected)
    );

    let other_log = make_tree_head("civ-log-v2", &log_a_entries);
    assert_eq!(
        compare_tree_heads(&head_a, &other_log),
        Err(ProofFailure::DifferentLog)
    );

    let shorter_head = make_tree_head("civ-log-v1", &log_a_entries[..3]);
    assert_eq!(
        compare_tree_heads(&head_a, &shorter_head),
        Err(ProofFailure::ConsistencyProofRequired)
    );

    println!("SYM-CIV-008 PASS: Merkle root, inclusion, consistency and fork smoke controls hold.");
    println!("Claim ceiling: local RFC 9162-shaped SHA-256 Merkle proofs only; no signed tree heads or live gossip.");
}
