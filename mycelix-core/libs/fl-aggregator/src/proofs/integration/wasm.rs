//! WebAssembly Bindings for Proof Verification
//!
//! Compile proofs to WASM for client-side verification in browsers.
//!
//! ## Features
//!
//! - Browser-compatible proof verification for all proof types
//! - Batch verification for multiple proofs
//! - Recursive proof verification
//! - JavaScript/TypeScript interop via wasm-bindgen
//! - Performance timing and console logging
//!
//! ## Building for WASM
//!
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build WASM package for bundlers (webpack, etc.)
//! wasm-pack build --target bundler --features proofs-wasm
//!
//! # Build for web (ESM)
//! wasm-pack build --target web --features proofs-wasm
//!
//! # Build for Node.js
//! wasm-pack build --target nodejs --features proofs-wasm
//! ```
//!
//! ## JavaScript/TypeScript Usage
//!
//! ```typescript
//! import init, {
//!   WasmProofVerifier,
//!   verify_range_proof,
//!   verify_gradient_proof,
//!   verify_identity_proof,
//!   verify_vote_proof,
//!   verify_batch,
//!   library_info,
//! } from 'fl-aggregator';
//!
//! // Initialize WASM module
//! await init();
//!
//! // Create a verifier instance
//! const verifier = new WasmProofVerifier();
//!
//! // Verify individual proofs
//! const rangeResult = verifier.verify_range(proofBytes, 0n, 100n, 42n);
//! console.log(`Valid: ${rangeResult.valid}, Time: ${rangeResult.verification_time_ms}ms`);
//!
//! // Verify gradient proof
//! const gradientResult = verify_gradient_proof(proofBytes, 5.0);
//! if (!gradientResult.valid) {
//!   console.error(`Error: ${gradientResult.error}`);
//! }
//!
//! // Batch verification
//! const batchResult = verify_batch([proof1, proof2, proof3]);
//! console.log(`${batchResult.valid_count}/${batchResult.total_count} valid`);
//!
//! // Get library info
//! const info = library_info();
//! console.log(`Version: ${info.version}, Proof types: ${info.proof_types}`);
//! ```

#[cfg(feature = "proofs-wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "proofs-wasm")]
use js_sys::{Array, Uint8Array};

use crate::proofs::{
    ProofType,
    RangeProof, MembershipProof, GradientIntegrityProof,
    IdentityAssuranceProof, VoteEligibilityProof,
};
use serde::{Deserialize, Serialize};

/// Performance timing helper
#[cfg(feature = "proofs-wasm")]
fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Console logging helper
#[cfg(feature = "proofs-wasm")]
fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

// ============================================================================
// Result Types
// ============================================================================

/// Verification result for JavaScript
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "proofs-wasm", wasm_bindgen)]
pub struct WasmVerificationResult {
    /// Whether the proof is valid
    pub valid: bool,
    /// Verification time in milliseconds
    pub verification_time_ms: f64,
    /// Error message if invalid
    error: Option<String>,
    /// Proof type that was verified
    proof_type: Option<String>,
}

#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
impl WasmVerificationResult {
    /// Get the error message if verification failed
    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }

    /// Get the proof type that was verified
    #[wasm_bindgen(getter)]
    pub fn proof_type(&self) -> Option<String> {
        self.proof_type.clone()
    }

    /// Create a successful result
    #[wasm_bindgen(constructor)]
    pub fn new(valid: bool, time_ms: f64, error: Option<String>) -> Self {
        Self {
            valid,
            verification_time_ms: time_ms,
            error,
            proof_type: None,
        }
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

impl WasmVerificationResult {
    pub fn success(time_ms: f64) -> Self {
        Self {
            valid: true,
            verification_time_ms: time_ms,
            error: None,
            proof_type: None,
        }
    }

    pub fn success_with_type(time_ms: f64, proof_type: ProofType) -> Self {
        Self {
            valid: true,
            verification_time_ms: time_ms,
            error: None,
            proof_type: Some(format!("{:?}", proof_type)),
        }
    }

    pub fn failure(error: String, time_ms: f64) -> Self {
        Self {
            valid: false,
            verification_time_ms: time_ms,
            error: Some(error),
            proof_type: None,
        }
    }

    pub fn failure_with_type(error: String, time_ms: f64, proof_type: ProofType) -> Self {
        Self {
            valid: false,
            verification_time_ms: time_ms,
            error: Some(error),
            proof_type: Some(format!("{:?}", proof_type)),
        }
    }
}

/// Batch verification result
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "proofs-wasm", wasm_bindgen)]
pub struct WasmBatchResult {
    /// Total number of proofs
    pub total_count: usize,
    /// Number of valid proofs
    pub valid_count: usize,
    /// Number of invalid proofs
    pub invalid_count: usize,
    /// Total verification time in milliseconds
    pub total_time_ms: f64,
    /// Average time per proof in milliseconds
    pub avg_time_ms: f64,
    /// Whether all proofs are valid
    pub all_valid: bool,
    /// Individual results as JSON
    results_json: String,
}

#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
impl WasmBatchResult {
    #[wasm_bindgen(getter)]
    pub fn results_json(&self) -> String {
        self.results_json.clone()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Envelope info for JavaScript
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "proofs-wasm", wasm_bindgen)]
pub struct WasmEnvelopeInfo {
    /// Proof type as string
    proof_type: String,
    /// Security level as string
    security_level: String,
    /// Proof size in bytes
    pub proof_size: usize,
    /// Whether the envelope is compressed
    pub compressed: bool,
    /// Version number
    pub version: u8,
    /// Timestamp if present
    pub timestamp_ms: Option<f64>,
}

#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
impl WasmEnvelopeInfo {
    #[wasm_bindgen(getter)]
    pub fn proof_type(&self) -> String {
        self.proof_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn security_level(&self) -> String {
        self.security_level.clone()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// Library information
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "proofs-wasm", wasm_bindgen)]
pub struct WasmLibraryInfo {
    /// Library version
    version: String,
    /// Supported proof types
    proof_types: String,
    /// Supported security levels
    security_levels: String,
    /// Feature flags enabled
    features: String,
}

#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
impl WasmLibraryInfo {
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.version.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn proof_types(&self) -> String {
        self.proof_types.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn security_levels(&self) -> String {
        self.security_levels.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn features(&self) -> String {
        self.features.clone()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// ============================================================================
// Main Verifier Class
// ============================================================================

/// WebAssembly Proof Verifier
///
/// Main entry point for browser-based proof verification.
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub struct WasmProofVerifier {
    /// Whether to log to console
    verbose: bool,
    /// Total proofs verified
    proofs_verified: usize,
    /// Total verification time
    total_time_ms: f64,
}

#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
impl WasmProofVerifier {
    /// Create a new verifier
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            verbose: false,
            proofs_verified: 0,
            total_time_ms: 0.0,
        }
    }

    /// Enable verbose logging
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Get statistics
    pub fn stats(&self) -> String {
        format!(
            "{{\"proofs_verified\":{},\"total_time_ms\":{:.2},\"avg_time_ms\":{:.2}}}",
            self.proofs_verified,
            self.total_time_ms,
            if self.proofs_verified > 0 {
                self.total_time_ms / self.proofs_verified as f64
            } else {
                0.0
            }
        )
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.proofs_verified = 0;
        self.total_time_ms = 0.0;
    }

    /// Verify a range proof
    pub fn verify_range(
        &mut self,
        proof_bytes: &[u8],
        min: u64,
        max: u64,
        expected_value: u64,
    ) -> WasmVerificationResult {
        let result = verify_range_proof(proof_bytes, min, max, expected_value);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Range proof verification: valid={}, time={:.2}ms",
                result.valid, result.verification_time_ms));
        }
        result
    }

    /// Verify a gradient integrity proof
    pub fn verify_gradient(
        &mut self,
        proof_bytes: &[u8],
        max_norm: f64,
    ) -> WasmVerificationResult {
        let result = verify_gradient_proof(proof_bytes, max_norm as f32);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Gradient proof verification: valid={}, time={:.2}ms",
                result.valid, result.verification_time_ms));
        }
        result
    }

    /// Verify an identity assurance proof
    pub fn verify_identity(
        &mut self,
        proof_bytes: &[u8],
        min_level: u8,
    ) -> WasmVerificationResult {
        let result = verify_identity_proof(proof_bytes, min_level);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Identity proof verification: valid={}, time={:.2}ms",
                result.valid, result.verification_time_ms));
        }
        result
    }

    /// Verify a vote eligibility proof
    pub fn verify_vote(
        &mut self,
        proof_bytes: &[u8],
        proposal_type: u8,
    ) -> WasmVerificationResult {
        let result = verify_vote_proof(proof_bytes, proposal_type);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Vote proof verification: valid={}, time={:.2}ms",
                result.valid, result.verification_time_ms));
        }
        result
    }

    /// Verify a membership proof
    pub fn verify_membership(
        &mut self,
        proof_bytes: &[u8],
        merkle_root: &[u8],
    ) -> WasmVerificationResult {
        let result = verify_membership_proof(proof_bytes, merkle_root);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Membership proof verification: valid={}, time={:.2}ms",
                result.valid, result.verification_time_ms));
        }
        result
    }

    /// Verify a proof envelope (auto-detects type)
    pub fn verify_envelope(&mut self, envelope_bytes: &[u8]) -> WasmVerificationResult {
        let result = verify_proof_envelope(envelope_bytes);
        self.proofs_verified += 1;
        self.total_time_ms += result.verification_time_ms;
        if self.verbose {
            log(&format!("Envelope proof verification: valid={}, type={:?}, time={:.2}ms",
                result.valid, result.proof_type, result.verification_time_ms));
        }
        result
    }
}

#[cfg(feature = "proofs-wasm")]
impl Default for WasmProofVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Individual Verification Functions
// ============================================================================

/// Verify a range proof from bytes
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_range_proof(
    proof_bytes: &[u8],
    min: u64,
    max: u64,
    _expected_value: u64,
) -> WasmVerificationResult {
    let start = now();

    let proof = match RangeProof::from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure_with_type(
                format!("Failed to deserialize: {}", e),
                now() - start,
                ProofType::Range,
            );
        }
    };

    let public_inputs = proof.public_inputs();
    if public_inputs.min != min || public_inputs.max != max {
        return WasmVerificationResult::failure_with_type(
            format!(
                "Public inputs mismatch: expected [{}, {}], got [{}, {}]",
                min, max, public_inputs.min, public_inputs.max
            ),
            now() - start,
            ProofType::Range,
        );
    }

    match proof.verify() {
        Ok(result) if result.is_valid => {
            WasmVerificationResult::success_with_type(now() - start, ProofType::Range)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            now() - start,
            ProofType::Range,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            now() - start,
            ProofType::Range,
        ),
    }
}

/// Verify a gradient integrity proof
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_gradient_proof(proof_bytes: &[u8], max_norm: f32) -> WasmVerificationResult {
    let start = now();

    let proof = match GradientIntegrityProof::from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure_with_type(
                format!("Failed to deserialize: {}", e),
                now() - start,
                ProofType::GradientIntegrity,
            );
        }
    };

    let public_inputs = proof.public_inputs();
    if (public_inputs.max_norm - max_norm).abs() > 0.01 {
        return WasmVerificationResult::failure_with_type(
            format!(
                "Max norm mismatch: expected {}, got {}",
                max_norm, public_inputs.max_norm
            ),
            now() - start,
            ProofType::GradientIntegrity,
        );
    }

    match proof.verify() {
        Ok(result) if result.is_valid => {
            WasmVerificationResult::success_with_type(now() - start, ProofType::GradientIntegrity)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            now() - start,
            ProofType::GradientIntegrity,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            now() - start,
            ProofType::GradientIntegrity,
        ),
    }
}

/// Verify an identity assurance proof
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_identity_proof(proof_bytes: &[u8], min_level: u8) -> WasmVerificationResult {
    let start = now();

    let proof = match IdentityAssuranceProof::from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure_with_type(
                format!("Failed to deserialize: {}", e),
                now() - start,
                ProofType::IdentityAssurance,
            );
        }
    };

    let public_inputs = proof.public_inputs();
    if public_inputs.min_level as u8 != min_level {
        return WasmVerificationResult::failure_with_type(
            format!(
                "Min level mismatch: expected E{}, got E{}",
                min_level, public_inputs.min_level as u8
            ),
            now() - start,
            ProofType::IdentityAssurance,
        );
    }

    match proof.verify() {
        Ok(result) if result.is_valid => {
            WasmVerificationResult::success_with_type(now() - start, ProofType::IdentityAssurance)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            now() - start,
            ProofType::IdentityAssurance,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            now() - start,
            ProofType::IdentityAssurance,
        ),
    }
}

/// Verify a vote eligibility proof
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_vote_proof(proof_bytes: &[u8], proposal_type: u8) -> WasmVerificationResult {
    use crate::proofs::ProofProposalType;

    let start = now();

    let proof = match VoteEligibilityProof::from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure_with_type(
                format!("Failed to deserialize: {}", e),
                now() - start,
                ProofType::VoteEligibility,
            );
        }
    };

    let expected_type = match proposal_type {
        0 => ProofProposalType::Standard,
        1 => ProofProposalType::Constitutional,
        2 => ProofProposalType::ModelGovernance,
        3 => ProofProposalType::Emergency,
        4 => ProofProposalType::Treasury,
        5 => ProofProposalType::Membership,
        _ => {
            return WasmVerificationResult::failure_with_type(
                format!("Invalid proposal type: {}", proposal_type),
                now() - start,
                ProofType::VoteEligibility,
            );
        }
    };

    let public_inputs = proof.public_inputs();
    if public_inputs.proposal_type != expected_type {
        return WasmVerificationResult::failure_with_type(
            format!(
                "Proposal type mismatch: expected {:?}, got {:?}",
                expected_type, public_inputs.proposal_type
            ),
            now() - start,
            ProofType::VoteEligibility,
        );
    }

    match proof.verify() {
        Ok(result) if result.is_valid => {
            WasmVerificationResult::success_with_type(now() - start, ProofType::VoteEligibility)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            now() - start,
            ProofType::VoteEligibility,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            now() - start,
            ProofType::VoteEligibility,
        ),
    }
}

/// Verify a membership proof
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_membership_proof(proof_bytes: &[u8], merkle_root: &[u8]) -> WasmVerificationResult {
    let start = now();

    let proof = match MembershipProof::from_bytes(proof_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure_with_type(
                format!("Failed to deserialize: {}", e),
                now() - start,
                ProofType::Membership,
            );
        }
    };

    // Check merkle root matches
    let public_inputs = proof.public_inputs();
    if merkle_root.len() != 32 {
        return WasmVerificationResult::failure_with_type(
            format!("Invalid merkle root length: expected 32, got {}", merkle_root.len()),
            now() - start,
            ProofType::Membership,
        );
    }

    let mut root_array = [0u8; 32];
    root_array.copy_from_slice(merkle_root);
    if public_inputs.merkle_root != root_array {
        return WasmVerificationResult::failure_with_type(
            "Merkle root mismatch".to_string(),
            now() - start,
            ProofType::Membership,
        );
    }

    match proof.verify() {
        Ok(result) if result.is_valid => {
            WasmVerificationResult::success_with_type(now() - start, ProofType::Membership)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            now() - start,
            ProofType::Membership,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            now() - start,
            ProofType::Membership,
        ),
    }
}

// ============================================================================
// Envelope Functions
// ============================================================================

/// Parse a proof envelope and get info
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn parse_proof_envelope(envelope_bytes: &[u8]) -> Result<WasmEnvelopeInfo, JsValue> {
    use super::serialization::ProofEnvelope;

    let envelope = ProofEnvelope::from_bytes(envelope_bytes)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse envelope: {}", e)))?;

    Ok(WasmEnvelopeInfo {
        proof_type: format!("{:?}", envelope.proof_type),
        security_level: format!("{:?}", envelope.security_level),
        proof_size: envelope.proof_bytes.len(),
        compressed: envelope.compressed,
        version: envelope.version,
        timestamp_ms: envelope.timestamp.map(|t| t as f64),
    })
}

/// Verify a proof envelope (auto-detects type)
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_proof_envelope(envelope_bytes: &[u8]) -> WasmVerificationResult {
    use super::serialization::ProofEnvelope;
    let start = now();

    let envelope = match ProofEnvelope::from_bytes(envelope_bytes) {
        Ok(e) => e,
        Err(e) => {
            return WasmVerificationResult::failure(
                format!("Failed to parse envelope: {}", e),
                now() - start,
            );
        }
    };

    let result = match envelope.proof_type {
        ProofType::Range => {
            RangeProof::from_bytes(&envelope.proof_bytes).and_then(|p| p.verify())
        }
        ProofType::GradientIntegrity => {
            GradientIntegrityProof::from_bytes(&envelope.proof_bytes).and_then(|p| p.verify())
        }
        ProofType::IdentityAssurance => {
            IdentityAssuranceProof::from_bytes(&envelope.proof_bytes).and_then(|p| p.verify())
        }
        ProofType::VoteEligibility => {
            VoteEligibilityProof::from_bytes(&envelope.proof_bytes).and_then(|p| p.verify())
        }
        ProofType::Membership => {
            MembershipProof::from_bytes(&envelope.proof_bytes).and_then(|p| p.verify())
        }
    };

    let elapsed = now() - start;
    match result {
        Ok(verification) if verification.is_valid => {
            WasmVerificationResult::success_with_type(elapsed, envelope.proof_type)
        }
        Ok(_) => WasmVerificationResult::failure_with_type(
            "Proof verification failed".to_string(),
            elapsed,
            envelope.proof_type,
        ),
        Err(e) => WasmVerificationResult::failure_with_type(
            format!("Verification error: {}", e),
            elapsed,
            envelope.proof_type,
        ),
    }
}

// ============================================================================
// Batch Verification
// ============================================================================

/// Verify a batch of proof envelopes
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_batch(envelopes: Vec<Uint8Array>) -> WasmBatchResult {
    let start = now();
    let total_count = envelopes.len();
    let mut valid_count = 0;
    let mut results = Vec::with_capacity(total_count);

    for envelope_bytes in envelopes {
        let bytes: Vec<u8> = envelope_bytes.to_vec();
        let result = verify_proof_envelope(&bytes);
        if result.valid {
            valid_count += 1;
        }
        results.push(result);
    }

    let total_time = now() - start;
    let results_json = serde_json::to_string(&results).unwrap_or_default();

    WasmBatchResult {
        total_count,
        valid_count,
        invalid_count: total_count - valid_count,
        total_time_ms: total_time,
        avg_time_ms: if total_count > 0 {
            total_time / total_count as f64
        } else {
            0.0
        },
        all_valid: valid_count == total_count,
        results_json,
    }
}

/// Verify recursive proof batch
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn verify_recursive_batch(batch_bytes: &[u8]) -> WasmVerificationResult {
    use crate::proofs::RecursiveProof;
    let start = now();

    let proof = match RecursiveProof::from_bytes(batch_bytes) {
        Ok(p) => p,
        Err(e) => {
            return WasmVerificationResult::failure(
                format!("Failed to deserialize recursive proof: {}", e),
                now() - start,
            );
        }
    };

    match proof.verify() {
        Ok(result) if result.all_valid => {
            let mut wasm_result = WasmVerificationResult::success(now() - start);
            wasm_result.proof_type = Some(format!("RecursiveBatch({})", result.proof_count));
            wasm_result
        }
        Ok(result) => {
            WasmVerificationResult::failure(
                format!("Batch verification failed: {}/{} valid", result.valid_count, result.proof_count),
                now() - start,
            )
        }
        Err(e) => WasmVerificationResult::failure(
            format!("Verification error: {}", e),
            now() - start,
        ),
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get supported proof types
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn supported_proof_types() -> Array {
    let arr = Array::new();
    arr.push(&JsValue::from_str("Range"));
    arr.push(&JsValue::from_str("GradientIntegrity"));
    arr.push(&JsValue::from_str("IdentityAssurance"));
    arr.push(&JsValue::from_str("VoteEligibility"));
    arr.push(&JsValue::from_str("Membership"));
    arr
}

/// Get supported security levels
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn supported_security_levels() -> Array {
    let arr = Array::new();
    arr.push(&JsValue::from_str("Standard96"));
    arr.push(&JsValue::from_str("Standard128"));
    arr.push(&JsValue::from_str("High256"));
    arr
}

/// Get library version
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn library_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get library info
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn library_info() -> WasmLibraryInfo {
    WasmLibraryInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        proof_types: "Range, GradientIntegrity, IdentityAssurance, VoteEligibility, Membership".to_string(),
        security_levels: "Standard96, Standard128, High256".to_string(),
        features: "verification, batch, recursive, envelope".to_string(),
    }
}

/// Hash data using BLAKE3
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn blake3_hash(data: &[u8]) -> Uint8Array {
    let hash = blake3::hash(data);
    Uint8Array::from(hash.as_bytes().as_slice())
}

/// Generate a commitment from data (hash with domain separation)
#[cfg(feature = "proofs-wasm")]
#[wasm_bindgen]
pub fn generate_commitment(domain: &str, data: &[u8]) -> Uint8Array {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(&[0u8]); // separator
    hasher.update(data);
    let hash = hasher.finalize();
    Uint8Array::from(hash.as_bytes().as_slice())
}

// ============================================================================
// Non-WASM implementations for testing
// ============================================================================

#[cfg(not(feature = "proofs-wasm"))]
impl WasmVerificationResult {
    pub fn success(time_ms: f64) -> Self {
        Self {
            valid: true,
            verification_time_ms: time_ms,
            error: None,
            proof_type: None,
        }
    }

    pub fn success_with_type(time_ms: f64, proof_type: ProofType) -> Self {
        Self {
            valid: true,
            verification_time_ms: time_ms,
            error: None,
            proof_type: Some(format!("{:?}", proof_type)),
        }
    }

    pub fn failure(error: String, time_ms: f64) -> Self {
        Self {
            valid: false,
            verification_time_ms: time_ms,
            error: Some(error),
            proof_type: None,
        }
    }

    pub fn failure_with_type(error: String, time_ms: f64, proof_type: ProofType) -> Self {
        Self {
            valid: false,
            verification_time_ms: time_ms,
            error: Some(error),
            proof_type: Some(format!("{:?}", proof_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_success() {
        let result = WasmVerificationResult::success(10.5);
        assert!(result.valid);
        assert_eq!(result.verification_time_ms, 10.5);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_verification_result_failure() {
        let result = WasmVerificationResult::failure("Test error".to_string(), 5.0);
        assert!(!result.valid);
        assert_eq!(result.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_verification_result_with_type() {
        let result = WasmVerificationResult::success_with_type(10.0, ProofType::Range);
        assert!(result.valid);
        assert_eq!(result.proof_type, Some("Range".to_string()));
    }

    #[test]
    fn test_library_info_struct() {
        let info = WasmLibraryInfo {
            version: "0.1.0".to_string(),
            proof_types: "Range".to_string(),
            security_levels: "Standard128".to_string(),
            features: "verification".to_string(),
        };
        assert_eq!(info.version, "0.1.0");
    }

    #[test]
    fn test_envelope_info_struct() {
        let info = WasmEnvelopeInfo {
            proof_type: "Range".to_string(),
            security_level: "Standard128".to_string(),
            proof_size: 1000,
            compressed: false,
            version: 1,
            timestamp_ms: Some(1234567890.0),
        };
        assert_eq!(info.proof_size, 1000);
        assert!(!info.compressed);
    }

    #[test]
    fn test_batch_result_struct() {
        let result = WasmBatchResult {
            total_count: 10,
            valid_count: 8,
            invalid_count: 2,
            total_time_ms: 100.0,
            avg_time_ms: 10.0,
            all_valid: false,
            results_json: "[]".to_string(),
        };
        assert_eq!(result.total_count, 10);
        assert_eq!(result.valid_count, 8);
        assert!(!result.all_valid);
    }
}
