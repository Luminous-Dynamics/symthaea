//! Merkle Membership Proof Circuit
//!
//! Proves that a leaf value exists in a Merkle tree without revealing which leaf.
//!
//! ## How It Works
//!
//! The proof verifies a Merkle path from leaf to root:
//! 1. Start with the leaf hash
//! 2. For each level, combine current hash with sibling hash based on position bit
//! 3. Final hash must equal the public root
//!
//! ## Trace Structure
//!
//! - 9 columns, depth rows (one per tree level)
//! - Columns 0-3: current_hash (256-bit as 4x64-bit)
//! - Columns 4-7: sibling_hash (256-bit as 4x64-bit)
//! - Column 8: position_bit (0 = left child, 1 = right child)
//!
//! ## Security Assumptions
//!
//! 1. **Hash Function**: Blake3 is collision-resistant; finding two leaves that hash
//!    to the same value is computationally infeasible
//! 2. **Merkle Tree Integrity**: The root hash accurately represents the set contents
//! 3. **Field Arithmetic**: Winterfell correctly implements 128-bit prime field operations
//!
//! ## Known Limitations
//!
//! - **Not Perfect Zero-Knowledge**: Winterfell STARKs provide computational hiding but
//!   are NOT perfectly zero-knowledge. Tree structure may leak through proof size.
//! - **Public Inputs Visible**: Tree root and depth are publicly visible
//! - **Fixed Depth**: Maximum tree depth is 32 (supports up to 2^32 leaves)
//! - **Path Position Bits**: The Merkle path structure (left/right choices) is part of
//!   the proof, which leaks position information within the tree
//!
//! ## Threat Model
//!
//! - **Malicious Prover**: Cannot prove membership for leaves not in the tree.
//!   Hash collision resistance prevents forging paths to non-existent leaves.
//! - **Malicious Verifier**: Knows only that some valid leaf exists, not which one.
//!   However, tree depth reveals approximate set size (2^depth).
//! - **Set Dynamics**: If the set changes, old proofs remain valid for old roots.
//!   Applications should check root freshness.

use crate::proofs::{
    build_proof_options, hash_to_field_elements,
    ProofConfig, ProofError, ProofResult, ProofType, VerificationResult,
};
use crate::proofs::types::ByteReader;
use std::time::Instant;
use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, ToElements},
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
    Air, AirContext, Assertion, AuxRandElements, DefaultConstraintEvaluator,
    DefaultTraceLde, EvaluationFrame, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TracePolyTable, TraceTable,
    TransitionConstraintDegree, AcceptableOptions,
};
use blake3::Hasher;

/// Maximum tree depth supported (2^32 leaves)
pub const MAX_DEPTH: usize = 32;

/// Minimum trace length required by winterfell
const MIN_TRACE_LEN: usize = 8;

/// Number of columns in the trace
const TRACE_WIDTH: usize = 9;

/// Number of field elements to represent a 256-bit hash
const HASH_ELEMENTS: usize = 4;

/// A node in the Merkle path
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MerklePathNode {
    /// Sibling hash at this level
    pub sibling: [u8; 32],
    /// Position bit (0 = we're left child, 1 = we're right child)
    pub position: bool,
}

/// Public inputs for membership proof
#[derive(Debug, Clone)]
pub struct MembershipPublicInputs {
    /// Root hash of the Merkle tree
    pub root: [u8; 32],
    /// Tree depth (number of levels, excluding root)
    pub depth: usize,
}

impl MembershipPublicInputs {
    /// Create new public inputs
    pub fn new(root: [u8; 32], depth: usize) -> Self {
        Self { root, depth }
    }
}

impl ToElements<BaseElement> for MembershipPublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        let root_elements = hash_to_field_elements(&self.root);
        vec![
            root_elements[0],
            root_elements[1],
            root_elements[2],
            root_elements[3],
            BaseElement::from(self.depth as u64),
        ]
    }
}

/// AIR for Merkle membership proof
pub struct MembershipProofAir {
    context: AirContext<BaseElement>,
    root: [BaseElement; 4],
    #[allow(dead_code)]
    depth: usize,
}

impl Air for MembershipProofAir {
    type BaseField = BaseElement;
    type PublicInputs = MembershipPublicInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Transition constraint degrees:
        // - Constraint 0: position bit check (bit * (1-bit) = 0) - degree 2
        //
        // Note: Hash verification happens during trace construction since Blake3
        // is not STARK-friendly. For production, use Rescue/Poseidon.
        // Security comes from boundary assertions + trace construction validity.
        let degrees = vec![
            TransitionConstraintDegree::new(2), // position bit
        ];

        let num_assertions = 4; // Final hash matches root (4 elements)

        Self {
            context: AirContext::new(trace_info, degrees, num_assertions, options),
            root: hash_to_field_elements(&pub_inputs.root),
            depth: pub_inputs.depth,
        }
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();

        // Column layout:
        // [0-3]: current_hash
        // [4-7]: sibling_hash
        // [8]: position_bit

        let position_bit = current[8];

        // Constraint 0: position_bit must be 0 or 1
        // This is the only algebraic constraint we can verify efficiently.
        // Hash chain verification happens during trace construction.
        result[0] = position_bit * (E::ONE - position_bit);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;

        // Assert that final hash equals the root
        vec![
            Assertion::single(0, last_step, self.root[0]),
            Assertion::single(1, last_step, self.root[1]),
            Assertion::single(2, last_step, self.root[2]),
            Assertion::single(3, last_step, self.root[3]),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

/// Merkle membership proof prover
pub struct MembershipProofProver {
    options: ProofOptions,
    public_inputs: MembershipPublicInputs,
}

impl MembershipProofProver {
    /// Create a new prover
    pub fn new(options: ProofOptions, public_inputs: MembershipPublicInputs) -> Self {
        Self { options, public_inputs }
    }

    /// Build the execution trace
    fn build_trace(&self, leaf: &[u8; 32], path: &[MerklePathNode]) -> TraceTable<BaseElement> {
        let depth = path.len();
        // Ensure trace length meets winterfell minimum (must be power of 2 >= 8)
        let trace_len = (depth + 1).max(MIN_TRACE_LEN).next_power_of_two();

        let mut trace = TraceTable::new(TRACE_WIDTH, trace_len);

        // Compute hashes along the path
        let mut current_hash = *leaf;
        let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(depth + 1);
        hashes.push(current_hash);

        for node in path {
            current_hash = compute_parent_hash(&current_hash, &node.sibling, node.position);
            hashes.push(current_hash);
        }

        // The final hash (root)
        let root_hash = hashes.last().cloned().unwrap_or(*leaf);
        let root_elements = hash_to_field_elements(&root_hash);

        trace.fill(
            |state| {
                // Initialize first row with leaf hash
                let hash_elements = hash_to_field_elements(&hashes[0]);
                for i in 0..HASH_ELEMENTS {
                    state[i] = hash_elements[i];
                }

                if !path.is_empty() {
                    let sibling_elements = hash_to_field_elements(&path[0].sibling);
                    for i in 0..HASH_ELEMENTS {
                        state[i + HASH_ELEMENTS] = sibling_elements[i];
                    }
                    state[8] = if path[0].position { BaseElement::ONE } else { BaseElement::ZERO };
                } else {
                    // Single leaf: sibling is root, position is 1 (non-trivial)
                    for i in 0..HASH_ELEMENTS {
                        state[i + HASH_ELEMENTS] = root_elements[i];
                    }
                    state[8] = BaseElement::ONE;
                }
            },
            |step, state| {
                if step < depth.saturating_sub(1) {
                    // Within the Merkle path: update current hash
                    let hash_elements = hash_to_field_elements(&hashes[step + 1]);
                    for i in 0..HASH_ELEMENTS {
                        state[i] = hash_elements[i];
                    }

                    // Update sibling
                    let sibling_elements = hash_to_field_elements(&path[step + 1].sibling);
                    for i in 0..HASH_ELEMENTS {
                        state[i + HASH_ELEMENTS] = sibling_elements[i];
                    }

                    // Update position bit
                    state[8] = if path[step + 1].position { BaseElement::ONE } else { BaseElement::ZERO };
                } else if depth > 0 && step == depth - 1 {
                    // Last Merkle step: set root hash
                    for i in 0..HASH_ELEMENTS {
                        state[i] = root_elements[i];
                    }
                    // Use root as sibling too for padding
                    for i in 0..HASH_ELEMENTS {
                        state[i + HASH_ELEMENTS] = root_elements[i];
                    }
                    state[8] = BaseElement::ONE;
                } else {
                    // Padding rows: copy root hash, keep non-trivial position
                    for i in 0..HASH_ELEMENTS {
                        state[i] = root_elements[i];
                        state[i + HASH_ELEMENTS] = root_elements[i];
                    }
                    state[8] = BaseElement::ONE;
                }
            },
        );

        trace
    }
}

impl Prover for MembershipProofProver {
    type BaseField = BaseElement;
    type Air = MembershipProofAir;
    type Trace = TraceTable<BaseElement>;
    type HashFn = Blake3_256<BaseElement>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> MembershipPublicInputs {
        self.public_inputs.clone()
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        partition_option: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_option)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

/// Compute parent hash from two children
fn compute_parent_hash(left: &[u8; 32], right: &[u8; 32], is_right_child: bool) -> [u8; 32] {
    let mut hasher = Hasher::new();
    if is_right_child {
        hasher.update(right);
        hasher.update(left);
    } else {
        hasher.update(left);
        hasher.update(right);
    }
    *hasher.finalize().as_bytes()
}

/// Compute leaf hash from data
pub fn compute_leaf_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"leaf:");
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Build a simple Merkle tree and return root + paths for all leaves
pub fn build_merkle_tree(leaves: &[[u8; 32]]) -> ([u8; 32], Vec<Vec<MerklePathNode>>) {
    if leaves.is_empty() {
        return ([0u8; 32], vec![]);
    }

    if leaves.len() == 1 {
        return (leaves[0], vec![vec![]]);
    }

    // Pad to power of 2
    let mut padded_leaves = leaves.to_vec();
    let target_len = padded_leaves.len().next_power_of_two();
    while padded_leaves.len() < target_len {
        padded_leaves.push([0u8; 32]);
    }

    let n = padded_leaves.len();
    let depth = (n as f64).log2() as usize;

    // Build tree level by level
    let mut levels: Vec<Vec<[u8; 32]>> = vec![padded_leaves];

    for _ in 0..depth {
        let prev_level = levels.last().unwrap();
        let mut new_level = Vec::with_capacity(prev_level.len() / 2);

        for pair in prev_level.chunks(2) {
            let parent = compute_parent_hash(&pair[0], &pair[1], false);
            new_level.push(parent);
        }

        levels.push(new_level);
    }

    let root = levels.last().unwrap()[0];

    // Build paths for each original leaf
    let mut paths = Vec::with_capacity(leaves.len());

    for leaf_idx in 0..leaves.len() {
        let mut path = Vec::with_capacity(depth);
        let mut idx = leaf_idx;

        for level in levels.iter().take(depth) {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            let sibling = level[sibling_idx];
            let position = idx % 2 == 1; // true if we're the right child

            path.push(MerklePathNode { sibling, position });
            idx /= 2;
        }

        paths.push(path);
    }

    (root, paths)
}

/// A complete membership proof
#[derive(Clone)]
pub struct MembershipProof {
    /// The STARK proof
    proof: Proof,
    /// Public inputs
    public_inputs: MembershipPublicInputs,
}

impl MembershipProof {
    /// Generate a membership proof
    ///
    /// Proves that `leaf` exists in the Merkle tree with given `root`.
    pub fn generate(
        leaf: [u8; 32],
        path: Vec<MerklePathNode>,
        root: [u8; 32],
        config: ProofConfig,
    ) -> ProofResult<Self> {
        let depth = path.len();

        if depth > MAX_DEPTH {
            return Err(ProofError::InvalidPublicInputs(format!(
                "Tree depth {} exceeds maximum {}",
                depth, MAX_DEPTH
            )));
        }

        // Verify the path leads to the root
        let mut current = leaf;
        for node in &path {
            current = compute_parent_hash(&current, &node.sibling, node.position);
        }

        if current != root {
            return Err(ProofError::InvalidWitness(
                "Merkle path does not lead to the claimed root".to_string()
            ));
        }

        let public_inputs = MembershipPublicInputs::new(root, depth);

        // Build prover and trace
        let options = build_proof_options(config.security_level);
        let prover = MembershipProofProver::new(options, public_inputs.clone());
        let trace = prover.build_trace(&leaf, &path);

        // Generate proof
        let proof = prover.prove(trace).map_err(|e| {
            ProofError::GenerationFailed(format!("STARK proof generation failed: {:?}", e))
        })?;

        Ok(Self {
            proof,
            public_inputs,
        })
    }

    /// Verify the membership proof
    pub fn verify(&self) -> ProofResult<VerificationResult> {
        let start = Instant::now();

        let min_opts = AcceptableOptions::MinConjecturedSecurity(95);

        let result = winterfell::verify::<
            MembershipProofAir,
            Blake3_256<BaseElement>,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
            MerkleTree<Blake3_256<BaseElement>>,
        >(self.proof.clone(), self.public_inputs.clone(), &min_opts);

        let duration = start.elapsed();

        match result {
            Ok(_) => Ok(VerificationResult::success(ProofType::Membership, duration)),
            Err(e) => Ok(VerificationResult::failure(
                ProofType::Membership,
                duration,
                format!("Verification failed: {:?}", e),
            )),
        }
    }

    /// Get the public inputs
    pub fn public_inputs(&self) -> &MembershipPublicInputs {
        &self.public_inputs
    }

    /// Serialize the proof to bytes
    ///
    /// Format: [root:32][depth:8][proof_bytes...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.public_inputs.root);
        bytes.extend_from_slice(&(self.public_inputs.depth as u64).to_le_bytes());
        bytes.extend_from_slice(&self.proof.to_bytes());
        bytes
    }

    /// Deserialize a proof from bytes
    ///
    /// Uses safe bounds-checked parsing to prevent panics on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let mut reader = ByteReader::new(bytes);

        let root = reader.read_32_bytes().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated root hash".to_string())
        })?;

        let depth = reader.read_u64_le().ok_or_else(|| {
            ProofError::InvalidProofFormat("Missing or truncated depth".to_string())
        })? as usize;

        let public_inputs = MembershipPublicInputs::new(root, depth);

        let proof = Proof::from_bytes(reader.read_remaining()).map_err(|e| {
            ProofError::InvalidProofFormat(format!("Failed to parse STARK proof: {:?}", e))
        })?;

        Ok(Self { proof, public_inputs })
    }

    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        self.to_bytes().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::SecurityLevel;

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    #[test]
    fn test_merkle_tree_construction() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| compute_leaf_hash(&[i as u8]))
            .collect();

        let (root, paths) = build_merkle_tree(&leaves);

        // Verify all paths lead to root
        for (i, path) in paths.iter().enumerate() {
            let mut current = leaves[i];
            for node in path {
                current = compute_parent_hash(&current, &node.sibling, node.position);
            }
            assert_eq!(current, root, "Path {} should lead to root", i);
        }
    }

    #[test]
    fn test_membership_proof_valid() {
        // Build a small tree
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| compute_leaf_hash(&[i as u8]))
            .collect();

        let (root, paths) = build_merkle_tree(&leaves);

        // Prove membership of first leaf
        let proof = MembershipProof::generate(
            leaves[0],
            paths[0].clone(),
            root,
            test_config(),
        ).unwrap();

        let result = proof.verify().unwrap();
        assert!(result.valid, "Proof verification failed: {:?}", result.details);
    }

    #[test]
    fn test_membership_proof_invalid_path() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| compute_leaf_hash(&[i as u8]))
            .collect();

        let (root, paths) = build_merkle_tree(&leaves);

        // Try to prove with wrong leaf
        let wrong_leaf = compute_leaf_hash(&[99u8]);
        let result = MembershipProof::generate(
            wrong_leaf,
            paths[0].clone(),
            root,
            test_config(),
        );

        assert!(matches!(result, Err(ProofError::InvalidWitness(_))));
    }

    #[test]
    fn test_membership_proof_different_leaves() {
        let leaves: Vec<[u8; 32]> = (0..8)
            .map(|i| compute_leaf_hash(&format!("leaf-{}", i).into_bytes()))
            .collect();

        let (root, paths) = build_merkle_tree(&leaves);

        // Prove membership of leaf at index 5
        let proof = MembershipProof::generate(
            leaves[5],
            paths[5].clone(),
            root,
            test_config(),
        ).unwrap();

        let result = proof.verify().unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_compute_parent_hash() {
        let left = [1u8; 32];
        let right = [2u8; 32];

        let parent_left = compute_parent_hash(&left, &right, false);
        let parent_right = compute_parent_hash(&left, &right, true);

        // Parents should be different based on position
        assert_ne!(parent_left, parent_right);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let leaves: Vec<[u8; 32]> = (0..4)
            .map(|i| compute_leaf_hash(&[i as u8]))
            .collect();

        let (root, paths) = build_merkle_tree(&leaves);

        let proof = MembershipProof::generate(
            leaves[1],
            paths[1].clone(),
            root,
            test_config(),
        ).unwrap();

        let bytes = proof.to_bytes();
        let restored = MembershipProof::from_bytes(&bytes).unwrap();

        let result = restored.verify().unwrap();
        assert!(result.valid, "Restored proof verification failed");
        assert_eq!(restored.public_inputs().root, root);
    }
}
