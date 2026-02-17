//! HyperFeel compressed gradient submission with zkSTARK proofs.

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

/// Submit a HyperFeel-compressed gradient with zkSTARK proof
///
/// This enhanced submission provides:
/// 1. 2000x bandwidth reduction via HyperFeel encoding
/// 2. Cryptographic proof of honest computation via zkSTARK
/// 3. Automatic trust scoring via PoGQ
/// 4. Byzantine detection with REJECTION of malicious gradients
#[hdk_extern]
pub fn submit_compressed_gradient(input: SubmitCompressedGradientInput) -> ExternResult<CompressedGradientResult> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;
    // F-06: Check rate limit
    check_rate_limit("submit_gradient")?;

    // Validate hypervector size
    if input.hypervector.len() != HV16_BYTES {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid hypervector size: expected {} bytes, got {}", HV16_BYTES, input.hypervector.len())
        )));
    }

    // Verify zkSTARK proof
    let proof_verified = verify_zkstark_proof(&input.proof_bytes, input.epochs);

    if !proof_verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "zkSTARK proof verification failed. Real proof verification is required for production. \
             Enable 'stub-proofs' feature for development/testing only.".to_string()
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
            "zkstark_proof_failed"
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

/// Verify zkSTARK proof - STUB version (development/testing only)
///
/// This stub only checks proof size and an epochs field at a fixed offset.
/// It provides ZERO actual cryptographic verification.
/// Enable the `stub-proofs` feature ONLY for development/testing.
#[cfg(feature = "stub-proofs")]
pub(crate) fn verify_zkstark_proof(proof_bytes: &[u8], epochs: u32) -> bool {
    // STUB: Only checks proof size and epochs field. NOT cryptographically secure.
    // Enable "stub-proofs" feature ONLY for development/testing.
    if proof_bytes.len() < 10_000 {
        return false;
    }

    // Check proof has valid structure (first 32 bytes should be commitment)
    if proof_bytes.len() >= 68 {
        // Extract encoded epochs and verify they match
        if let Ok(epochs_bytes) = <[u8; 4]>::try_from(&proof_bytes[64..68]) {
            let encoded_epochs = u32::from_le_bytes(epochs_bytes);
            if encoded_epochs != epochs {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

/// Verify zkSTARK proof - production version (always rejects until real verification is implemented)
///
/// Real zkSTARK verification is not yet implemented. All proof submissions are
/// rejected until a genuine cryptographic verifier is integrated.
#[cfg(not(feature = "stub-proofs"))]
pub(crate) fn verify_zkstark_proof(_proof_bytes: &[u8], _epochs: u32) -> bool {
    // Real zkSTARK verification not yet implemented.
    // All proof submissions are rejected until real verification is available.
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
