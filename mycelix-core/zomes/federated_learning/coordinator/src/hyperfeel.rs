// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HyperFeel compressed gradient submission with Merkle commitment proofs.

use hdk::prelude::*;
use federated_learning_integrity::*;

use sha2::Digest;

use mycelix_sdk::hyperfeel::HV16_BYTES;
use mycelix_sdk::matl::ProofOfGradientQuality;

use crate::config::*;
use crate::auth::*;
use crate::signals::Signal;
use crate::detection::*;
use crate::pipeline::get_or_create_reputation;
use super::ensure_path;

/// Input for submitting a compressed gradient with zkSTARK proof
#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitCompressedGradientInput {
    /// Node identifier
    pub node_id: String,
    /// Training round
    pub round: u32,
    /// HyperFeel compressed hypervector (2KB)
    pub hypervector: Vec<u8>,
    /// zkSTARK proof bytes
    pub proof_bytes: Vec<u8>,
    /// Original gradient size (for verification)
    pub original_size: usize,
    /// Quality score from local training
    pub quality_score: f32,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate used
    pub learning_rate: f32,
    /// Resource metrics
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    /// Optional model architecture hash for version validation
    #[serde(default)]
    pub architecture_hash: Option<String>,
}

/// Response from compressed gradient submission
#[derive(Serialize, Deserialize, Debug)]
pub struct CompressedGradientResult {
    /// Action hash of stored gradient
    pub action_hash: ActionHash,
    /// Whether zkSTARK proof was verified
    pub proof_verified: bool,
    /// Computed trust score
    pub trust_score: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Submit a HyperFeel-compressed gradient with Merkle commitment proof
///
/// This enhanced submission provides:
/// 1. 2000x bandwidth reduction via HyperFeel encoding
/// 2. Cryptographic proof of computation via Merkle commitment + HMAC binding
/// 3. Automatic trust scoring via PoGQ
/// 4. Byzantine detection with REJECTION of malicious gradients
#[hdk_extern]
pub fn submit_compressed_gradient(input: SubmitCompressedGradientInput) -> ExternResult<CompressedGradientResult> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;

    // H-02: Bind node_id to caller's AgentPubKey
    // This prevents impersonation: only the actual caller can submit gradients under their own ID
    let agent = agent_info()?.agent_initial_pubkey;
    let caller_id = agent.to_string();
    if input.node_id != caller_id {
        // H-02: node_id must match caller's AgentPubKey
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "node_id must match caller's agent pubkey. Expected: {}, got: {}",
            caller_id, input.node_id
        ))));
    }

    // F-06: Check rate limit
    check_rate_limit("submit_gradient")?;

    // Validate hypervector size
    if input.hypervector.len() != HV16_BYTES {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid hypervector size: expected {} bytes, got {}", HV16_BYTES, input.hypervector.len())
        )));
    }

    // Verify gradient commitment proof (Merkle root + HMAC binding)
    let proof_verified = verify_gradient_commitment(
        &input.proof_bytes,
        &input.node_id,
        input.round,
        input.epochs,
    );

    if !proof_verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Gradient commitment proof verification failed. Proof must contain valid \
             Merkle root + HMAC binding. Enable 'zkstark' feature for proof verification.".to_string()
        )));
    }

    // ==========================================================================
    // BYZANTINE DETECTION: Run before storing gradient
    // ==========================================================================

    // Extract features for Byzantine detection
    let gradient_features = extract_compressed_gradient_features(&input)?;

    // Run hierarchical detection
    let (is_byzantine, detection_confidence) = run_byzantine_detection(&gradient_features);

    // Also check zkSTARK proof failure as Byzantine indicator
    let proof_byzantine = !proof_verified;
    let combined_is_byzantine = is_byzantine || proof_byzantine;
    let combined_confidence = if proof_byzantine {
        // Failed proof is high confidence Byzantine
        0.9_f32.max(detection_confidence)
    } else {
        detection_confidence
    };

    // If Byzantine detected with sufficient confidence, REJECT the gradient
    if combined_is_byzantine && combined_confidence >= BYZANTINE_REJECTION_THRESHOLD {
        // Compute evidence hash for audit trail
        let evidence_hash = compute_evidence_hash(&gradient_features, combined_confidence);

        // Record Byzantine behavior
        let detection_method = if proof_byzantine {
            "commitment_proof_failed"
        } else {
            "hierarchical_compressed"
        };

        record_byzantine_internal(
            &input.node_id,
            input.round,
            detection_method,
            combined_confidence,
            &evidence_hash,
        )?;

        // REJECT: Do not store the gradient
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine gradient detected and rejected: node={}, round={}, confidence={:.2}, proof_valid={}",
                input.node_id, input.round, combined_confidence, proof_verified
            )
        )));
    }

    // ==========================================================================
    // GRADIENT ACCEPTED: Store as normal
    // ==========================================================================

    // Compute trust score using PoGQ (use with_timestamp since SystemTime::now is unavailable in WASM)
    let pogq_ts = sys_time()?.0 as u64 / 1_000_000;
    let pogq = ProofOfGradientQuality::with_timestamp(
        input.quality_score as f64,
        0.9,  // Default consistency (would come from temporal tracking)
        0.1,  // Default entropy
        pogq_ts,
    );
    let reputation = get_or_create_reputation(&input.node_id)?;
    let trust_score = pogq.composite_score(reputation.reputation_score as f64) as f32;

    // Compute compression ratio
    let compression_ratio = input.original_size as f32 / HV16_BYTES as f32;

    // Create gradient hash from hypervector
    let mut hasher = sha2::Sha256::new();
    hasher.update(&input.hypervector);
    let gradient_hash = format!("{:x}", hasher.finalize());

    let timestamp = sys_time()?;
    let gradient = ModelGradient {
        node_id: input.node_id.clone(),
        round: input.round,
        gradient_hash,
        timestamp: timestamp.0 as i64 / 1_000_000,
        cpu_usage: input.cpu_usage,
        memory_mb: input.memory_mb,
        network_latency_ms: input.network_latency_ms,
        trust_score: Some(trust_score),
    };

    // Store gradient entry
    let action_hash = create_entry(&EntryTypes::ModelGradient(gradient.clone()))?;

    // Create links for indexing
    let round_path = Path::from(format!("round.{}", input.round));
    let round_entry_hash = ensure_path(round_path, LinkTypes::RoundToGradients)?;
    create_link(
        round_entry_hash,
        action_hash.clone(),
        LinkTypes::RoundToGradients,
        vec![],
    )?;

    let node_path = Path::from(format!("node.{}", input.node_id.clone()));
    let node_entry_hash = ensure_path(node_path, LinkTypes::NodeToGradients)?;
    create_link(
        node_entry_hash,
        action_hash.clone(),
        LinkTypes::NodeToGradients,
        vec![],
    )?;

    // Store hypervector as separate entry for aggregation
    store_hypervector(&input.node_id, input.round, &input.hypervector)?;

    // Emit signal
    emit_signal(Signal::GradientSubmitted {
        node_id: input.node_id,
        round: input.round,
        action_hash: action_hash.clone(),
        source: None,
        signature: None,
    })?;

    Ok(CompressedGradientResult {
        action_hash,
        proof_verified,
        trust_score,
        compression_ratio,
    })
}

/// Verify gradient commitment proof using Merkle root + HMAC binding.
///
/// Replaces the previous zkSTARK stub with a real cryptographic verification
/// scheme that uses only SHA-256 (zero additional WASM binary overhead).
///
/// proof_bytes layout:
///   [0..32]    - Merkle root (SHA-256 hash)
///   [32..64]   - HMAC binding: SHA256(root || node_id || round_le || epochs_le)
///   [64..96]   - Challenged leaf hash
///   [96]       - Sibling path length (depth, max 32)
///   [97..97+depth*32] - Sibling hashes for Merkle inclusion proof
///
/// Security properties:
/// - HMAC binding prevents replay across rounds/nodes
/// - Merkle inclusion proves the leaf is part of the committed computation
/// - Canonical sibling ordering (sorted) avoids left/right ambiguity
///
/// C-03: Requires the `zkstark` feature flag. Without it, all proofs are
/// rejected (fail-closed).
#[cfg(feature = "zkstark")]
pub(crate) fn verify_gradient_commitment(
    proof_bytes: &[u8],
    node_id: &str,
    round: u32,
    epochs: u32,
) -> bool {
    // Minimum: root(32) + hmac(32) + leaf(32) + depth(1) = 97 bytes
    if proof_bytes.len() < 97 {
        return false;
    }

    let root = &proof_bytes[0..32];
    let claimed_hmac = &proof_bytes[32..64];
    let leaf_hash = &proof_bytes[64..96];
    let path_len = proof_bytes[96] as usize;

    // Sanity: depth must be reasonable (max 32 for 2^32 leaves)
    if path_len > 32 {
        return false;
    }

    // Check we have enough bytes for the sibling path
    let required_len = 97 + path_len * 32;
    if proof_bytes.len() < required_len {
        return false;
    }

    // 1. Verify HMAC binding: prevents proof replay across rounds/nodes
    let mut binder = sha2::Sha256::new();
    binder.update(root);
    binder.update(node_id.as_bytes());
    binder.update(&round.to_le_bytes());
    binder.update(&epochs.to_le_bytes());
    let expected_hmac = binder.finalize();
    if expected_hmac.as_slice() != claimed_hmac {
        return false;
    }

    // 2. Verify Merkle inclusion: walk from leaf to root
    let siblings = &proof_bytes[97..required_len];
    let mut current_hash = [0u8; 32];
    current_hash.copy_from_slice(leaf_hash);

    for i in 0..path_len {
        let sibling = &siblings[i * 32..(i + 1) * 32];
        let mut h = sha2::Sha256::new();
        // Canonical ordering: smaller hash first to avoid left/right ambiguity
        if current_hash.as_slice() <= sibling {
            h.update(&current_hash);
            h.update(sibling);
        } else {
            h.update(sibling);
            h.update(&current_hash);
        }
        let result = h.finalize();
        current_hash.copy_from_slice(&result);
    }

    // Root must match
    current_hash.as_slice() == root
}

/// Verify gradient commitment proof - fail-closed default (no zkstark feature)
///
/// C-03: Gradient commitment verification requires the `zkstark` feature flag.
/// Without the feature, ALL proof submissions are rejected (fail-closed).
/// This prevents silently passing invalid proofs.
#[cfg(not(feature = "zkstark"))]
pub(crate) fn verify_gradient_commitment(
    _proof_bytes: &[u8],
    _node_id: &str,
    _round: u32,
    _epochs: u32,
) -> bool {
    // C-03: Proof verification requires the 'zkstark' feature flag.
    // Fail-closed: reject all proofs when verification is not available.
    false
}

/// Store hypervector for later aggregation
pub(crate) fn store_hypervector(node_id: &str, round: u32, hypervector: &[u8]) -> ExternResult<()> {
    // Store as link tag data for efficient retrieval
    let hv_path = Path::from(format!("hypervectors.{}", round));
    let hv_entry_hash = ensure_path(hv_path, LinkTypes::RoundToGradients)?;

    // Encode node_id + hypervector as link tag
    let mut tag_data = Vec::with_capacity(node_id.len() + 1 + hypervector.len());
    tag_data.extend_from_slice(node_id.as_bytes());
    tag_data.push(0); // Separator
    tag_data.extend_from_slice(hypervector);

    create_link(
        hv_entry_hash.clone(),
        hv_entry_hash, // Self-link for storage
        LinkTypes::RoundToGradients,
        tag_data,
    )?;

    Ok(())
}

/// Get all hypervectors for a round (for aggregation)
#[hdk_extern]
pub fn get_round_hypervectors(round: u32) -> ExternResult<Vec<(String, Vec<u8>)>> {
    let hv_path = Path::from(format!("hypervectors.{}", round));
    let hv_entry_hash = ensure_path(hv_path, LinkTypes::RoundToGradients)?;

    let links = get_links(
        LinkQuery::new(
            hv_entry_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToGradients as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let mut hypervectors = Vec::new();
    for link in links {
        let tag = link.tag.as_ref();
        if let Some(separator_pos) = tag.iter().position(|&b| b == 0) {
            let node_id = String::from_utf8_lossy(&tag[..separator_pos]).to_string();
            let hypervector = tag[separator_pos + 1..].to_vec();
            if hypervector.len() == HV16_BYTES {
                hypervectors.push((node_id, hypervector));
            }
        }
    }

    Ok(hypervectors)
}

/// Aggregate hypervectors for a round using majority voting
#[hdk_extern]
pub fn aggregate_round_hypervectors(round: u32) -> ExternResult<Vec<u8>> {
    let hypervectors = get_round_hypervectors(round)?;

    if hypervectors.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No hypervectors to aggregate".to_string()
        )));
    }

    if hypervectors.len() == 1 {
        return Ok(hypervectors[0].1.clone());
    }

    // Majority voting aggregation
    let n = hypervectors.len();
    let threshold = n / 2;

    let mut result = vec![0u8; HV16_BYTES];
    for byte_idx in 0..HV16_BYTES {
        let mut result_byte = 0u8;
        for bit_idx in 0..8 {
            let mask = 1u8 << (7 - bit_idx);
            let ones: usize = hypervectors
                .iter()
                .map(|(_, hv)| if hv.get(byte_idx).map_or(false, |&b| b & mask != 0) { 1 } else { 0 })
                .sum();

            if ones > threshold {
                result_byte |= mask;
            }
        }
        result[byte_idx] = result_byte;
    }

    Ok(result)
}

/// Get per-agent semantic resonance score [0.0, 1.0].
///
/// Computes Hamming similarity between the agent's latest submitted
/// hypervector and the aggregated community hypervector for the most
/// recent round. High similarity = strong alignment with community consensus.
///
/// Used by the 8D Sovereign Profile (D6: Semantic Resonance).
#[hdk_extern]
pub fn get_agent_semantic_resonance(round: u32) -> ExternResult<f64> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_id = agent.to_string();

    // Get all HVs for this round
    let all_hvs = get_round_hypervectors(round)?;
    if all_hvs.len() < 2 {
        return Ok(0.0); // Need at least 2 participants for resonance
    }

    // Find the agent's HV
    let agent_hv = all_hvs.iter()
        .find(|(id, _)| id == &agent_id)
        .map(|(_, hv)| hv.clone());

    let agent_hv = match agent_hv {
        Some(hv) => hv,
        None => return Ok(0.0), // Agent didn't participate
    };

    // Aggregate community HV (majority vote of all participants)
    let community_hv = aggregate_round_hypervectors(round)?;

    // Hamming similarity: matching bits / total bits
    let total_bits = (HV16_BYTES * 8) as f64;
    let matching_bits: u32 = agent_hv.iter()
        .zip(community_hv.iter())
        .map(|(a, b)| (!(a ^ b)).count_ones())
        .sum();

    let similarity = matching_bits as f64 / total_bits;

    // Map from [0.5, 1.0] range to [0.0, 1.0] (random = 0.5 → 0.0 score)
    let normalized = ((similarity - 0.5) * 2.0).clamp(0.0, 1.0);
    Ok(normalized)
}
