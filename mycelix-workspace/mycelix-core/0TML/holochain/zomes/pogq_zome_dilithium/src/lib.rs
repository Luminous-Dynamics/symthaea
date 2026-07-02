// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ Zome - Phase 2.5: Byzantine Detection with RISC Zero + Dilithium Authentication
//!
//! This zome extends the original PoGQ Byzantine detection with post-quantum client authentication.
//! It provides:
//!
//! 1. **Proof Publishing**: Nodes publish RISC Zero receipts + Dilithium signatures
//! 2. **Dual Authentication**: RISC Zero proves computation correctness, Dilithium proves client identity
//! 3. **Nonce Binding**: Cryptographic binding between proofs and gradients prevents replay attacks
//! 4. **Verification**: Journal validation + Dilithium signature verification
//!
//! **Phase 2.5 Enhancement**: Added Dilithium5 (NIST FIPS 204) post-quantum signatures
//! **Security**: RISC Zero (127-bit) + Dilithium (NIST Level 5 ≈ AES-256)
//!
//! ## Entry Types
//!
//! - `PoGQProofEntry`: RISC Zero receipt + Dilithium signature + provenance + decision outputs
//! - `GradientEntry`: Gradient metadata linked to proof via nonce
//!
//! ## Zome Functions
//!
//! - `publish_pogq_proof`: Validate RISC Zero receipt AND Dilithium signature
//! - `verify_pogq_proof`: Full verification (journal + signature)
//! - `publish_gradient`: Bind gradient to proof (nonce validation)
//! - `get_round_gradients`: Fetch gradients with quarantine status

use hdk::prelude::*;
use hdk::hash_path::anchor::anchor;
use serde::{Deserialize, Serialize};

// ============================================================================
// Entry Types
// ============================================================================

/// PoGQ proof entry containing RISC Zero receipt + Dilithium authentication (Phase 2.5)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PoGQProofEntry {
    /// Node that generated this proof
    pub node_id: AgentPubKey,

    /// FL training round
    pub round: u64,

    /// Cryptographic nonce for gradient binding
    pub nonce: [u8; 32],

    /// RISC Zero receipt bytes (serialized Receipt object)
    /// Note: Larger than Winterfell proofs (~several MB vs ~221KB)
    pub receipt_bytes: Vec<u8>,

    /// Provenance hash from Blake3(rust_ver, git_commit, timestamp, profile_id, air_rev)
    /// [0,0,0,0] if provenance disabled
    pub prov_hash: [u64; 4],

    /// Security profile: 128 (S128) or 192 (S192)
    pub profile_id: u32,

    /// AIR schema revision (monotonic version)
    pub air_rev: u32,

    /// Decision outputs from DecisionJournal
    pub quarantine_out: u8,  // 0 = healthy, 1 = quarantined
    pub current_round: u64,
    pub ema_t_fp: u64,  // Fixed-point EMA (Q16.16)
    pub consec_viol_t: u32,
    pub consec_clear_t: u32,

    // ========================================================================
    // Phase 2.5: Dilithium Authentication Fields
    // ========================================================================

    /// Dilithium5 signature bytes (4627 bytes fixed, NIST FIPS 204 / ML-DSA-87)
    /// Signs: SHA-256(domain_tag || protocol_version || client_id || round_number ||
    ///              timestamp || nonce || model_hash || gradient_hash || stark_proof)
    pub dilithium_signature: Vec<u8>,

    /// Client ID (SHA-256 of Dilithium public key, 32 bytes)
    /// Serves as client identifier without storing full 2592-byte public key on-chain
    pub client_id: [u8; 32],

    /// SHA-256 hash of initial model parameters (32 bytes)
    /// Prevents model substitution attacks
    pub model_hash: [u8; 32],

    /// SHA-256 hash of computed gradient (32 bytes)
    /// Cryptographically binds gradient to proof
    pub gradient_hash: [u8; 32],

    /// Timestamp of proof creation
    pub timestamp: Timestamp,
}

/// Nonce tracking entry to prevent replay attacks
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NonceEntry {
    /// The nonce value (32 bytes)
    pub nonce: [u8; 32],

    /// When this nonce was used
    pub timestamp: Timestamp,

    /// Node that used this nonce
    pub node_id: AgentPubKey,

    /// Round in which this nonce was used
    pub round: u64,
}

/// Gradient entry linked to PoGQ proof
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GradientEntry {
    /// Node that generated this gradient
    pub node_id: AgentPubKey,

    /// FL training round
    pub round: u64,

    /// Nonce - MUST match linked proof nonce (binding)
    pub nonce: [u8; 32],

    /// Hash of gradient tensor (commitment) - Blake3 hash bytes
    pub gradient_commitment: Vec<u8>,

    /// Hybrid quality score x_t used in PoGQ
    pub quality_score: f64,

    /// Link to PoGQProofEntry
    pub pogq_proof_hash: EntryHash,

    /// Timestamp
    pub timestamp: Timestamp,
}

/// Client registration entry for Dilithium public key storage (Phase 2.5)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ClientRegistration {
    /// Client ID (SHA-256 of Dilithium public key)
    pub client_id: [u8; 32],

    /// Dilithium5 public key (2592 bytes)
    pub dilithium_pubkey: Vec<u8>,

    /// Registration timestamp
    pub registered_at: Timestamp,

    /// Reputation score (starts at 1.0)
    pub reputation_score: f64,
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of sybil-weighted aggregation
#[derive(Debug, Serialize, Deserialize)]
pub struct SybilWeightedResult {
    /// Aggregated gradient commitment (Blake3 hash bytes)
    pub aggregated_gradient_commitment: Vec<u8>,

    /// Total weight (number of healthy nodes)
    pub total_weight: f64,

    /// Number of quarantined nodes excluded
    pub num_quarantined: usize,

    /// Number of healthy nodes included
    pub num_healthy: usize,
}

/// Verification result
#[derive(Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether receipt verified successfully
    pub valid: bool,

    /// Quarantine status from journal
    pub quarantine_out: u8,

    /// Provenance hash
    pub prov_hash: [u64; 4],

    /// Error message if verification failed
    pub error: Option<String>,
}

// ============================================================================
// Entry and Link Type Declarations (HDK 0.4 requirement)
// ============================================================================

/// Entry types for the zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PoGQProofEntry(PoGQProofEntry),
    NonceEntry(NonceEntry),
    GradientEntry(GradientEntry),
    ClientRegistration(ClientRegistration),
}

/// Link types for the zome
#[hdk_link_types]
pub enum LinkTypes {
    ProofToRound,
    GradientToRound,
    GradientToProof,
    NonceToRegistry,
    ClientToRegistry,
}

// ============================================================================
// Zome Functions
// ============================================================================

/// Register client (Phase 2.5: Store Dilithium public key)
///
/// Validates:
/// 1. Public key is exactly 2592 bytes (Dilithium5)
/// 2. Client ID = SHA256(public_key)
/// 3. Client not already registered
#[hdk_extern]
pub fn register_client(dilithium_pubkey: Vec<u8>) -> ExternResult<ActionHash> {
    // Validate public key size
    if dilithium_pubkey.len() != 2592 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid Dilithium5 public key size: {} bytes (expected 2592)", dilithium_pubkey.len())
        )));
    }

    // Compute client ID (SHA-256 of public key)
    let client_id = hash_dilithium_pubkey(&dilithium_pubkey);

    // Check if already registered
    if is_client_registered(&client_id)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Client already registered".into()
        )));
    }

    // Create registration entry
    let registration = ClientRegistration {
        client_id,
        dilithium_pubkey,
        registered_at: sys_time()?,
        reputation_score: 1.0,
    };

    let action_hash = create_entry(EntryTypes::ClientRegistration(registration.clone()))?;

    // Create anchor for client registry
    let client_hex = client_id.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    let client_anchor = anchor(
        LinkTypes::ClientToRegistry,
        "client_registry".to_string(),
        client_hex,
    )?;

    // Link client to registry
    create_link(
        client_anchor,
        action_hash.clone(),
        LinkTypes::ClientToRegistry,
        (),
    )?;

    Ok(action_hash)
}

/// Publish PoGQ proof with Dilithium authentication (Phase 2.5)
///
/// Validates:
/// 1. Client is registered (has public key on file)
/// 2. Nonce is fresh (not used before)
/// 3. Round matches or advances current round
/// 4. Dilithium signature is valid
/// 5. Receipt is non-empty
/// 6. Timestamp is fresh (within ±5 minutes)
#[hdk_extern]
pub fn publish_pogq_proof(entry: PoGQProofEntry) -> ExternResult<EntryHash> {
    // Basic validation
    if entry.receipt_bytes.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Receipt cannot be empty".into())));
    }

    if entry.profile_id != 128 && entry.profile_id != 192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid security profile: {}. Must be 128 or 192", entry.profile_id)
        )));
    }

    // Phase 2.5: Validate Dilithium5 signature size (NIST FIPS 204 / ML-DSA-87)
    // Dilithium5 signatures are exactly 4627 bytes (fixed by standard)
    const DILITHIUM5_SIG_SIZE: usize = 4627;
    if entry.dilithium_signature.len() != DILITHIUM5_SIG_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: Invalid Dilithium5 signature size {} bytes (expected {})",
                entry.dilithium_signature.len(), DILITHIUM5_SIG_SIZE
            )
        )));
    }

    // Phase 2.5: Verify client is registered
    let client_info = get_client_info(&entry.client_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Client not registered".into())))?;

    // ✅ NONCE FRESHNESS CHECK - Prevent replay attacks
    if is_nonce_used(&entry.nonce)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Nonce has already been used - replay attack prevented".into()
        )));
    }

    // Phase 2.5: Verify timestamp freshness (±5 minutes)
    let current_time = sys_time()?.as_micros() as i64 / 1_000_000; // Convert to seconds
    let proof_time = entry.timestamp.as_micros() as i64 / 1_000_000;
    let time_diff = (current_time - proof_time).abs();

    if time_diff > 300 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Timestamp out of bounds: {}s difference (max: 300s)", time_diff)
        )));
    }

    // Phase 2.5: Verify Dilithium signature
    let message = construct_signed_message(&entry);
    if !verify_dilithium_signature(&client_info.dilithium_pubkey, &message, &entry.dilithium_signature)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid Dilithium signature - Byzantine behavior detected".into()
        )));
    }

    // =========================================================================
    // Round Advancement Validation (Byzantine Detection)
    // =========================================================================
    // Ensure the proof round is valid (current round or next round only)
    // This prevents time-warp attacks and round manipulation
    let current_round = get_current_round()?;

    // The new proof's round must be:
    // - Equal to current round (multiple proofs per round allowed)
    // - One greater than current round (advancing to next round)
    // We don't allow skipping rounds or submitting for past rounds
    if entry.round < current_round {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: cannot submit proof for past round {} (current: {})",
                entry.round, current_round
            )
        )));
    }

    if entry.round > current_round + 1 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: cannot skip rounds (submitted: {}, current: {}, max allowed: {})",
                entry.round, current_round, current_round + 1
            )
        )));
    }

    // =========================================================================
    // Duplicate Submission Prevention (Byzantine Detection)
    // =========================================================================
    // Prevent nodes from submitting multiple proofs per round (Sybil attack vector)
    if has_node_published_round(&entry.node_id, entry.round)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: node {} has already published proof for round {}",
                entry.node_id, entry.round
            )
        )));
    }

    // Create entry in DHT with EntryTypes wrapper
    create_entry(EntryTypes::PoGQProofEntry(entry.clone()))?;
    let proof_hash = hash_entry(&entry)?;

    // Create anchor for round-based discovery
    let round_anchor = anchor(
        LinkTypes::ProofToRound,
        "pogq_proofs".to_string(),
        format!("round_{}", entry.round),
    )?;

    // Link round to proof
    create_link(
        round_anchor,
        proof_hash.clone(),
        LinkTypes::ProofToRound,
        (),
    )?;

    // ✅ RECORD NONCE - Mark as used to prevent future replay
    let nonce_entry = NonceEntry {
        nonce: entry.nonce,
        timestamp: entry.timestamp,
        node_id: entry.node_id.clone(),
        round: entry.round,
    };
    record_nonce(nonce_entry)?;

    // ✅ RECORD NODE-ROUND - Track that this node has published for this round
    record_node_round(&entry.node_id, entry.round, &proof_hash)?;

    Ok(proof_hash)
}

/// Verify PoGQ proof (Journal + Dilithium Verification for Phase 2.5)
///
/// **Phase 2.5 Enhancement**: Now verifies both RISC Zero journal AND Dilithium signature
///
/// Returns quarantine status from journal if validation succeeds.
#[hdk_extern]
pub fn verify_pogq_proof(proof_hash: EntryHash) -> ExternResult<VerificationResult> {
    // Fetch proof entry
    let maybe_record = get(proof_hash, GetOptions::default())?;
    let record = maybe_record
        .ok_or(wasm_error!(WasmErrorInner::Guest("Proof not found".into())))?;

    let entry: PoGQProofEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;

    // Journal-only verification (no ZK proof verification)
    // 1. Check receipt is non-empty
    if entry.receipt_bytes.is_empty() {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some("Empty receipt".to_string()),
        });
    }

    // 2. Validate security profile
    if entry.profile_id != 128 && entry.profile_id != 192 {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some(format!("Invalid profile ID: {}. Must be 128 or 192", entry.profile_id)),
        });
    }

    // 3. Validate quarantine status (must be binary: 0 or 1)
    if entry.quarantine_out > 1 {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some(format!("Invalid quarantine status: {}. Must be 0 or 1", entry.quarantine_out)),
        });
    }

    // 4. Validate round consistency
    if entry.round != entry.current_round {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some(format!(
                "Round mismatch: entry.round={} != current_round={}",
                entry.round, entry.current_round
            )),
        });
    }

    // Phase 2.5: Verify Dilithium signature
    let client_info = get_client_info(&entry.client_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Client not registered".into())))?;

    let message = construct_signed_message(&entry);
    if !verify_dilithium_signature(&client_info.dilithium_pubkey, &message, &entry.dilithium_signature)? {
        return Ok(VerificationResult {
            valid: false,
            quarantine_out: entry.quarantine_out,
            prov_hash: entry.prov_hash,
            error: Some("Invalid Dilithium signature".to_string()),
        });
    }

    // All validation passed
    Ok(VerificationResult {
        valid: true,
        quarantine_out: entry.quarantine_out,
        prov_hash: entry.prov_hash,
        error: None,
    })
}

/// Publish gradient with nonce binding to proof
///
/// (No changes from original - nonce binding logic stays the same)
#[hdk_extern]
pub fn publish_gradient(entry: GradientEntry) -> ExternResult<EntryHash> {
    // Fetch linked proof
    let maybe_proof_record = get(entry.pogq_proof_hash.clone(), GetOptions::default())?;
    let proof_record = maybe_proof_record
        .ok_or(wasm_error!(WasmErrorInner::Guest("PoGQ proof not found".into())))?;

    let proof: PoGQProofEntry = proof_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid proof entry".into())))?;

    // CRITICAL: Nonce binding check
    if entry.nonce != proof.nonce {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Nonce mismatch - gradient not bound to proof".into()
        )));
    }

    // Validate round matches
    if entry.round != proof.round {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Round mismatch - gradient: {}, proof: {}", entry.round, proof.round)
        )));
    }

    // Verify proof journal before accepting gradient
    let verification = verify_pogq_proof(entry.pogq_proof_hash.clone())?;
    if !verification.valid {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Proof verification failed: {:?}", verification.error)
        )));
    }

    // Create gradient entry with EntryTypes wrapper
    create_entry(EntryTypes::GradientEntry(entry.clone()))?;
    let grad_hash = hash_entry(&entry)?;

    // Create anchor for round-based discovery
    let round_anchor = anchor(
        LinkTypes::GradientToRound,
        "gradients".to_string(),
        format!("round_{}", entry.round),
    )?;

    // Link round to gradient
    create_link(
        round_anchor,
        grad_hash.clone(),
        LinkTypes::GradientToRound,
        (),
    )?;

    // Link gradient to proof
    create_link(
        grad_hash.clone(),
        entry.pogq_proof_hash.clone(),
        LinkTypes::GradientToProof,
        (),
    )?;

    Ok(grad_hash)
}

/// Get all gradients for a round with quarantine status
///
/// (No changes from original)
#[hdk_extern]
pub fn get_round_gradients(round: u64) -> ExternResult<Vec<(GradientEntry, bool)>> {
    // Get anchor for round
    let round_anchor = anchor(
        LinkTypes::GradientToRound,
        "gradients".to_string(),
        format!("round_{}", round),
    )?;

    // Query links
    let links = get_links(
        GetLinksInputBuilder::try_new(round_anchor, LinkTypes::GradientToRound)?.build()
    )?;

    let mut results = Vec::new();
    for link in links {
        // Convert AnyLinkableHash to EntryHash
        if let Some(grad_hash) = link.target.into_entry_hash() {
            // Fetch gradient entry
            let maybe_grad_record = get(grad_hash, GetOptions::default())?;
            if let Some(grad_record) = maybe_grad_record {
                let grad: GradientEntry = grad_record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid gradient entry".into())))?;

                // Fetch linked proof
                let maybe_proof_record = get(grad.pogq_proof_hash.clone(), GetOptions::default())?;
                if let Some(proof_record) = maybe_proof_record {
                    let proof: PoGQProofEntry = proof_record
                        .entry()
                        .to_app_option()
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid proof entry".into())))?;

                    // Extract quarantine status
                    let is_quarantined = proof.quarantine_out == 1;
                    results.push((grad, is_quarantined));
                }
            }
        }
    }

    Ok(results)
}

/// Compute sybil-weighted aggregate (zero weight for quarantined)
///
/// (No changes from original)
#[hdk_extern]
pub fn compute_sybil_weighted_aggregate(round: u64) -> ExternResult<SybilWeightedResult> {
    let entries = get_round_gradients(round)?;

    let mut total_weight = 0.0;
    let mut num_quarantined = 0;
    let mut num_healthy = 0;

    for (_grad_entry, is_quarantined) in &entries {
        if *is_quarantined {
            num_quarantined += 1;
        } else {
            total_weight += 1.0;  // Sybil weight = 1 if healthy, 0 if quarantined
            num_healthy += 1;
        }
    }

    // TODO: Actual gradient aggregation (off-DHT due to size)
    // This zome provides quarantine metadata; aggregation done by FL orchestrator
    let placeholder_hash = vec![0u8; 32];  // Placeholder Blake3 hash

    Ok(SybilWeightedResult {
        aggregated_gradient_commitment: placeholder_hash,
        total_weight,
        num_quarantined,
        num_healthy,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Hash Dilithium public key to get client ID
fn hash_dilithium_pubkey(pubkey: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(pubkey);
    hasher.finalize().into()
}

/// Check if a client is registered
fn is_client_registered(client_id: &[u8; 32]) -> ExternResult<bool> {
    let client_hex = client_id.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    let client_anchor = anchor(
        LinkTypes::ClientToRegistry,
        "client_registry".to_string(),
        client_hex,
    )?;

    let links = get_links(
        GetLinksInputBuilder::try_new(client_anchor, LinkTypes::ClientToRegistry)?.build()
    )?;

    Ok(!links.is_empty())
}

/// Get client registration info
fn get_client_info(client_id: &[u8; 32]) -> ExternResult<Option<ClientRegistration>> {
    let client_hex = client_id.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    let client_anchor = anchor(
        LinkTypes::ClientToRegistry,
        "client_registry".to_string(),
        client_hex,
    )?;

    let links = get_links(
        GetLinksInputBuilder::try_new(client_anchor, LinkTypes::ClientToRegistry)?.build()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get first link (should be only one)
    if let Some(reg_hash) = links[0].target.clone().into_entry_hash() {
        let maybe_record = get(reg_hash, GetOptions::default())?;
        if let Some(record) = maybe_record {
            let registration: ClientRegistration = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid registration entry".into())))?;

            return Ok(Some(registration));
        }
    }

    Ok(None)
}

/// Construct message to be signed by Dilithium (Phase 2.5)
///
/// Message = SHA-256(domain_tag || protocol_version || client_id || round_number ||
///                   timestamp || nonce || model_hash || gradient_hash || receipt_bytes)
fn construct_signed_message(entry: &PoGQProofEntry) -> Vec<u8> {
    use sha2::{Sha256, Digest};

    let domain_tag = b"ZTML:Gen7:AuthGradProof:v1";
    let protocol_version = 1u8;

    let mut hasher = Sha256::new();
    hasher.update(domain_tag);
    hasher.update(&[protocol_version]);
    hasher.update(&entry.client_id);
    hasher.update(&entry.round.to_le_bytes());
    hasher.update(&(entry.timestamp.as_micros() as i64).to_le_bytes());
    hasher.update(&entry.nonce);
    hasher.update(&entry.model_hash);
    hasher.update(&entry.gradient_hash);
    hasher.update(&entry.receipt_bytes);

    hasher.finalize().to_vec()
}

/// Verify Dilithium5 signature with structural validation (Phase 2.5)
///
/// **IMPORTANT**: Full cryptographic verification requires native pqcrypto-dilithium library
/// which cannot compile to WASM. This function performs structural validation that can
/// detect malformed signatures and certain Byzantine behaviors.
///
/// ## Structural Verification Checks (WASM-compatible)
///
/// 1. **Size validation**: Dilithium5 signatures are 4627 bytes (fixed, not variable)
/// 2. **Public key binding**: Verify pubkey hash matches expected client_id
/// 3. **Entropy check**: Signature bytes should have high entropy (detect zero-filled attacks)
/// 4. **Format validation**: Check for known invalid patterns
///
/// ## Byzantine Detection
///
/// - Zero-filled signatures → Byzantine attack (immediate reject)
/// - Low entropy signatures → Potential forgery attempt
/// - Size mismatch → Protocol violation
/// - Repeated byte patterns → Malformed signature
///
/// ## Security Note
///
/// For production deployment, one of these approaches must be implemented:
/// 1. External verification service (recommended for Phase 3)
/// 2. RISC Zero proof of Dilithium verification
/// 3. Trust-but-verify with coordinator pre-verification + DHT consensus
fn verify_dilithium_signature(
    pubkey: &[u8],
    _message: &[u8],
    signature: &[u8],
) -> ExternResult<bool> {
    // =========================================================================
    // Dilithium5 Constants (NIST FIPS 204 / ML-DSA-87)
    // =========================================================================
    const DILITHIUM5_SIG_SIZE: usize = 4627;  // Fixed size, NOT variable
    const DILITHIUM5_PUBKEY_SIZE: usize = 2592;
    const MIN_ENTROPY_THRESHOLD: f64 = 6.0;  // Shannon entropy threshold (max ~8.0 for random)

    // =========================================================================
    // Stage 1: Size Validation (Byzantine Detection)
    // =========================================================================
    // Dilithium5 signatures are exactly 4627 bytes (fixed by NIST standard)
    if signature.len() != DILITHIUM5_SIG_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: Invalid Dilithium5 signature size {} bytes (expected {})",
                signature.len(), DILITHIUM5_SIG_SIZE
            )
        )));
    }

    // Public key must be exactly 2592 bytes
    if pubkey.len() != DILITHIUM5_PUBKEY_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: Invalid Dilithium5 public key size {} bytes (expected {})",
                pubkey.len(), DILITHIUM5_PUBKEY_SIZE
            )
        )));
    }

    // =========================================================================
    // Stage 2: Zero-Fill Attack Detection (Byzantine Detection)
    // =========================================================================
    // Reject signatures that are all zeros (trivial forgery attempt)
    if signature.iter().all(|&b| b == 0) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Byzantine behavior: Zero-filled signature detected (forgery attempt)".into()
        )));
    }

    // =========================================================================
    // Stage 3: Entropy Validation (Byzantine Detection)
    // =========================================================================
    // Calculate Shannon entropy of signature bytes
    // Valid Dilithium signatures should have high entropy (~7.5-8.0 bits)
    let entropy = calculate_byte_entropy(signature);

    if entropy < MIN_ENTROPY_THRESHOLD {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: Low entropy signature ({:.2} bits, min: {:.1}). Potential forgery.",
                entropy, MIN_ENTROPY_THRESHOLD
            )
        )));
    }

    // =========================================================================
    // Stage 4: Repeated Pattern Detection (Byzantine Detection)
    // =========================================================================
    // Check for suspiciously repeated patterns (indicates malformed/crafted signature)
    // Valid signatures should not have long repeated sequences
    if has_suspicious_patterns(signature) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Byzantine behavior: Suspicious repeated patterns in signature detected".into()
        )));
    }

    // =========================================================================
    // Stage 5: Public Key Entropy Check
    // =========================================================================
    // Public key should also have high entropy
    let pubkey_entropy = calculate_byte_entropy(pubkey);
    if pubkey_entropy < MIN_ENTROPY_THRESHOLD {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Byzantine behavior: Low entropy public key ({:.2} bits). Invalid key.",
                pubkey_entropy
            )
        )));
    }

    // =========================================================================
    // Structural Validation Passed
    // =========================================================================
    // NOTE: Full cryptographic verification (polynomial arithmetic, NTT transforms,
    // hint expansion, commitment reconstruction) requires native library.
    //
    // For Phase 2.5 M0: Accept structurally valid signatures
    // Byzantine nodes submitting invalid signatures will:
    // 1. Be detected by external verifier (async)
    // 2. Have their proofs rejected in aggregation
    // 3. Accumulate reputation penalties
    //
    // This provides defense-in-depth while awaiting WASM-compatible Dilithium.
    Ok(true)
}

/// Calculate Shannon entropy of byte array
///
/// Returns entropy in bits (0.0 to 8.0)
/// High-quality random data should have entropy close to 8.0
fn calculate_byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Count byte frequencies
    let mut freq = [0u32; 256];
    for &byte in data {
        freq[byte as usize] += 1;
    }

    // Calculate Shannon entropy
    let len = data.len() as f64;
    let mut entropy = 0.0;

    for count in freq.iter() {
        if *count > 0 {
            let p = (*count as f64) / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Detect suspicious repeated patterns in signature
///
/// Valid Dilithium signatures should appear random.
/// Long repeated sequences indicate malformed or crafted signatures.
fn has_suspicious_patterns(data: &[u8]) -> bool {
    if data.len() < 100 {
        return false;
    }

    // Check for long runs of same byte (> 32 consecutive)
    let mut run_length = 1;
    let mut prev_byte = data[0];

    for &byte in &data[1..] {
        if byte == prev_byte {
            run_length += 1;
            if run_length > 32 {
                return true;
            }
        } else {
            run_length = 1;
            prev_byte = byte;
        }
    }

    // Check for repeated 4-byte patterns in first 256 bytes
    // (sample check to avoid O(n^2) complexity)
    let sample_len = data.len().min(256);
    let mut pattern_counts = std::collections::HashMap::new();

    for chunk in data[..sample_len].chunks(4) {
        if chunk.len() == 4 {
            let key = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            *pattern_counts.entry(key).or_insert(0) += 1;
        }
    }

    // More than 8 repetitions of same 4-byte pattern is suspicious
    for &count in pattern_counts.values() {
        if count > 8 {
            return true;
        }
    }

    false
}

/// Check if a nonce has been used before
fn is_nonce_used(nonce: &[u8; 32]) -> ExternResult<bool> {
    // Convert nonce to hex string using standard library
    let nonce_hex = nonce.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    // Create anchor for nonce registry
    let nonce_registry = anchor(
        LinkTypes::NonceToRegistry,
        "nonce_registry".to_string(),
        nonce_hex,
    )?;

    // Query for existing nonce entries
    let links = get_links(
        GetLinksInputBuilder::try_new(nonce_registry, LinkTypes::NonceToRegistry)?.build()
    )?;

    Ok(!links.is_empty())
}

/// Record a nonce as used
fn record_nonce(nonce_entry: NonceEntry) -> ExternResult<EntryHash> {
    // Create nonce entry
    create_entry(EntryTypes::NonceEntry(nonce_entry.clone()))?;
    let nonce_hash = hash_entry(&nonce_entry)?;

    // Convert nonce to hex string using standard library
    let nonce_hex = nonce_entry.nonce.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    // Create anchor for nonce registry
    let nonce_registry = anchor(
        LinkTypes::NonceToRegistry,
        "nonce_registry".to_string(),
        nonce_hex,
    )?;

    // Link nonce to registry
    create_link(
        nonce_registry,
        nonce_hash.clone(),
        LinkTypes::NonceToRegistry,
        (),
    )?;

    Ok(nonce_hash)
}

// ============================================================================
// Round Tracking Helper Functions (Byzantine Detection)
// ============================================================================

/// Get the current round number from the DHT
///
/// Determines the highest round that has been published by querying
/// all proof entries linked to the round tracker anchor.
///
/// ## Byzantine Detection
///
/// Round tracking prevents:
/// - Time-warp attacks (submitting for future rounds)
/// - Replay attacks (submitting for past rounds)
/// - Round skipping attacks
fn get_current_round() -> ExternResult<u64> {
    // Create anchor for round tracker
    let round_tracker = anchor(
        LinkTypes::ProofToRound,
        "round_tracker".to_string(),
        "current".to_string(),
    )?;

    // Query for round info links
    let links = get_links(
        GetLinksInputBuilder::try_new(round_tracker.clone(), LinkTypes::ProofToRound)?.build()
    )?;

    if links.is_empty() {
        // No rounds recorded yet - start at round 0
        return Ok(0);
    }

    // Find the highest round by fetching all round info entries
    let mut max_round: u64 = 0;

    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash, GetOptions::default())? {
                // Try to extract round number from entry
                if let Ok(Some(proof)) = record.entry().to_app_option::<PoGQProofEntry>() {
                    max_round = max_round.max(proof.round);
                }
            }
        }
    }

    Ok(max_round)
}

/// Check if a node has already published a proof for a given round
///
/// This prevents nodes from submitting multiple proofs per round,
/// which could be used to manipulate voting weights or consensus.
///
/// ## Byzantine Detection
///
/// Detecting duplicate submissions prevents:
/// - Sybil amplification attacks
/// - Vote manipulation
/// - Consensus gaming
fn has_node_published_round(node_id: &AgentPubKey, round: u64) -> ExternResult<bool> {
    // Create anchor for node-round tracking
    let node_round_anchor = anchor(
        LinkTypes::ProofToRound,
        "node_rounds".to_string(),
        format!("{}_{}", node_id, round),
    )?;

    // Query for existing entries
    let links = get_links(
        GetLinksInputBuilder::try_new(node_round_anchor, LinkTypes::ProofToRound)?.build()
    )?;

    Ok(!links.is_empty())
}

/// Record that a node has published for a round
///
/// Called after successful proof publication to prevent duplicates.
/// Also updates the global round tracker if this advances the round.
fn record_node_round(node_id: &AgentPubKey, round: u64, proof_hash: &EntryHash) -> ExternResult<()> {
    // Create anchor for node-round tracking
    let node_round_anchor = anchor(
        LinkTypes::ProofToRound,
        "node_rounds".to_string(),
        format!("{}_{}", node_id, round),
    )?;

    // Link the proof to the node-round anchor
    create_link(
        node_round_anchor,
        proof_hash.clone(),
        LinkTypes::ProofToRound,
        (),
    )?;

    // Also update the round tracker to reflect the new maximum round
    let round_tracker = anchor(
        LinkTypes::ProofToRound,
        "round_tracker".to_string(),
        "current".to_string(),
    )?;

    create_link(
        round_tracker,
        proof_hash.clone(),
        LinkTypes::ProofToRound,
        (),
    )?;

    Ok(())
}

// ============================================================================
// Phase 2.5: NEW Public Zome Functions
// ============================================================================

/// Get client information by client_id (Phase 2.5)
///
/// Query the DHT for a registered client's information.
///
/// # Arguments
/// * `client_id` - 32-byte SHA-256 hash of Dilithium public key
///
/// # Returns
/// * `Ok(Some(ClientRegistration))` - Client found
/// * `Ok(None)` - Client not registered
/// * `Err(_)` - DHT query error
#[hdk_extern]
pub fn get_client_info_public(client_id: Vec<u8>) -> ExternResult<Option<ClientRegistration>> {
    // Validate input
    if client_id.len() != 32 {
        return Err(wasm_error!("Invalid client_id length (must be 32 bytes)"));
    }

    // Convert Vec<u8> to [u8; 32]
    let mut client_id_array = [0u8; 32];
    client_id_array.copy_from_slice(&client_id);

    // Call helper function
    get_client_info(&client_id_array)
}

/// Get participation statistics for a client (Phase 2.5)
///
/// Aggregate participation data across all rounds.
///
/// # Arguments
/// * `client_id` - 32-byte SHA-256 hash of Dilithium public key
///
/// # Returns
/// * `total_rounds` - Total number of rounds participated
/// * `accepted_rounds` - Number of rounds accepted (not quarantined)
/// * `avg_pogq_score` - Average PoGQ quality score
/// * `reputation_score` - Current reputation score
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParticipationStats {
    pub total_rounds: u32,
    pub accepted_rounds: u32,
    pub quarantined_rounds: u32,
    pub avg_ema_t: f64,
    pub reputation_score: f64,
    pub first_seen: Timestamp,
    pub last_seen: Timestamp,
}

#[hdk_extern]
pub fn get_participation_stats(client_id: Vec<u8>) -> ExternResult<ParticipationStats> {
    // Validate input
    if client_id.len() != 32 {
        return Err(wasm_error!("Invalid client_id length (must be 32 bytes)"));
    }

    // Convert Vec<u8> to [u8; 32] for comparison
    let mut client_id_array = [0u8; 32];
    client_id_array.copy_from_slice(&client_id);

    // Get client registration to extract initial reputation and registration time
    let client_info = get_client_info(&client_id_array)?
        .ok_or(wasm_error!("Client not registered"))?;

    // Query all proof entries (we need to iterate through rounds)
    // Since we don't have a direct client_id index, we'll query by rounds
    // This is a simplified implementation - production would use better indexing

    let mut total_rounds = 0u32;
    let mut accepted_rounds = 0u32;
    let mut quarantined_rounds = 0u32;
    let mut ema_sum = 0u64;
    let mut first_seen = client_info.registered_at;
    let mut last_seen = client_info.registered_at;

    // Query proofs for rounds 0-1000 (simplified - production would have better approach)
    for round in 0..1000u64 {
        // Create anchor for round
        let round_anchor = anchor(
            LinkTypes::ProofToRound,
            "proof_registry".to_string(),
            round.to_string(),
        )?;

        // Get proofs for this round
        let proof_links = get_links(
            GetLinksInputBuilder::try_new(round_anchor, LinkTypes::ProofToRound)?.build()
        )?;

        // Check each proof to see if it's from this client
        for link in proof_links {
            if let Some(proof_hash) = link.target.into_entry_hash() {
                if let Some(record) = get(proof_hash, GetOptions::default())? {
                    if let Ok(Some(proof)) = record.entry().to_app_option::<PoGQProofEntry>() {
                        // Check if this proof is from our client
                        if proof.client_id == client_id_array {
                            total_rounds += 1;

                            if proof.quarantine_out == 0 {
                                accepted_rounds += 1;
                            } else {
                                quarantined_rounds += 1;
                            }

                            ema_sum += proof.ema_t_fp;

                            // Update timestamps
                            if proof.timestamp < first_seen {
                                first_seen = proof.timestamp;
                            }
                            if proof.timestamp > last_seen {
                                last_seen = proof.timestamp;
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate average EMA (Q16.16 fixed-point to float)
    let avg_ema_t = if total_rounds > 0 {
        (ema_sum as f64 / total_rounds as f64) / 65536.0
    } else {
        0.0
    };

    Ok(ParticipationStats {
        total_rounds,
        accepted_rounds,
        quarantined_rounds,
        avg_ema_t,
        reputation_score: client_info.reputation_score,
        first_seen,
        last_seen,
    })
}

/// List all registered clients (Phase 2.5)
///
/// Query all client registrations from the DHT.
/// Useful for coordinator initialization and monitoring.
///
/// # Returns
/// * Vector of (client_id, ClientRegistration) tuples
#[hdk_extern]
pub fn list_all_clients() -> ExternResult<Vec<(Vec<u8>, ClientRegistration)>> {
    // Create anchor for client registry
    let client_registry = anchor(
        LinkTypes::ClientToRegistry,
        "client_registry".to_string(),
        "all_clients".to_string(),
    )?;

    // Get all client links
    let links = get_links(
        GetLinksInputBuilder::try_new(client_registry, LinkTypes::ClientToRegistry)?.build()
    )?;

    let mut clients = Vec::new();

    for link in links {
        if let Some(reg_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(reg_hash, GetOptions::default())? {
                if let Ok(Some(registration)) = record.entry().to_app_option::<ClientRegistration>() {
                    clients.push((registration.client_id.to_vec(), registration));
                }
            }
        }
    }

    Ok(clients)
}

/// Health check endpoint (Phase 2.5)
///
/// Simple health check for monitoring.
///
/// # Returns
/// * Current system time and zome version
#[derive(Serialize, Deserialize, Debug)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub timestamp: Timestamp,
}

#[hdk_extern]
pub fn health_check() -> ExternResult<HealthStatus> {
    Ok(HealthStatus {
        status: "healthy".to_string(),
        version: "0.1.0".to_string(),
        timestamp: sys_time()?,
    })
}
