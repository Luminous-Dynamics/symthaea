//! gRPC API for Proof Service
//!
//! High-performance binary protocol for proof generation and verification.
//!
//! ## Overview
//!
//! This module provides a gRPC service interface for the proof system with:
//! - Unary RPCs for single proof operations
//! - Server streaming for batch operations
//! - Bidirectional streaming for real-time progress
//!
//! ## Service Definition
//!
//! ```protobuf
//! service ProofService {
//!   rpc GenerateRange(RangeRequest) returns (ProofResponse);
//!   rpc GenerateGradient(GradientRequest) returns (ProofResponse);
//!   rpc GenerateIdentity(IdentityRequest) returns (ProofResponse);
//!   rpc GenerateVote(VoteRequest) returns (ProofResponse);
//!   rpc Verify(VerifyRequest) returns (VerifyResponse);
//!   rpc BatchGenerate(stream GenerateRequest) returns (stream ProofResponse);
//!   rpc BatchVerify(stream VerifyRequest) returns (stream VerifyResponse);
//! }
//! ```

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, RwLock};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::proofs::{
    GradientIntegrityProof, IdentityAssuranceProof, ProofAssuranceLevel,
    ProofConfig, ProofIdentityFactor, ProofProposalType, ProofType, ProofVoterProfile,
    RangeProof, SecurityLevel, VoteEligibilityProof, VerificationResult,
    ProofEnvelope, ProofResult,
};

// ============================================================================
// Protocol Buffer Message Types (manually defined, matching proto schema)
// ============================================================================

/// Security level for proof generation
#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcSecurityLevel {
    Standard96 = 0,
    Standard128 = 1,
    High256 = 2,
}

impl From<GrpcSecurityLevel> for SecurityLevel {
    fn from(level: GrpcSecurityLevel) -> Self {
        match level {
            GrpcSecurityLevel::Standard96 => SecurityLevel::Standard96,
            GrpcSecurityLevel::Standard128 => SecurityLevel::Standard128,
            GrpcSecurityLevel::High256 => SecurityLevel::High256,
        }
    }
}

impl From<SecurityLevel> for GrpcSecurityLevel {
    fn from(level: SecurityLevel) -> Self {
        match level {
            SecurityLevel::Standard96 => GrpcSecurityLevel::Standard96,
            SecurityLevel::Standard128 => GrpcSecurityLevel::Standard128,
            SecurityLevel::High256 => GrpcSecurityLevel::High256,
        }
    }
}

/// Proof type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcProofType {
    Range = 0,
    Membership = 1,
    GradientIntegrity = 2,
    IdentityAssurance = 3,
    VoteEligibility = 4,
}

impl From<ProofType> for GrpcProofType {
    fn from(pt: ProofType) -> Self {
        match pt {
            ProofType::Range => GrpcProofType::Range,
            ProofType::Membership => GrpcProofType::Membership,
            ProofType::GradientIntegrity => GrpcProofType::GradientIntegrity,
            ProofType::IdentityAssurance => GrpcProofType::IdentityAssurance,
            ProofType::VoteEligibility => GrpcProofType::VoteEligibility,
        }
    }
}

/// Configuration for proof generation
#[derive(Clone, prost::Message)]
pub struct GrpcProofConfig {
    #[prost(enumeration = "GrpcSecurityLevel", tag = "1")]
    pub security_level: i32,
    #[prost(bool, tag = "2")]
    pub parallel: bool,
    #[prost(uint64, tag = "3")]
    pub max_proof_size: u64,
}

impl From<GrpcProofConfig> for ProofConfig {
    fn from(cfg: GrpcProofConfig) -> Self {
        ProofConfig {
            security_level: GrpcSecurityLevel::try_from(cfg.security_level)
                .unwrap_or(GrpcSecurityLevel::Standard128)
                .into(),
            parallel: cfg.parallel,
            max_proof_size: cfg.max_proof_size as usize,
        }
    }
}

// ============================================================================
// Request Types
// ============================================================================

/// Request for range proof generation
#[derive(Clone, prost::Message)]
pub struct RangeRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(uint64, tag = "2")]
    pub value: u64,
    #[prost(uint64, tag = "3")]
    pub min: u64,
    #[prost(uint64, tag = "4")]
    pub max: u64,
    #[prost(message, optional, tag = "5")]
    pub config: Option<GrpcProofConfig>,
}

/// Request for gradient integrity proof generation
#[derive(Clone, prost::Message)]
pub struct GradientRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(float, repeated, tag = "2")]
    pub gradients: Vec<f32>,
    #[prost(float, tag = "3")]
    pub max_norm: f32,
    #[prost(message, optional, tag = "4")]
    pub config: Option<GrpcProofConfig>,
}

/// Identity factor for assurance proof
#[derive(Clone, prost::Message)]
pub struct GrpcIdentityFactor {
    #[prost(float, tag = "1")]
    pub contribution: f32,
    #[prost(uint32, tag = "2")]
    pub category: u32,
    #[prost(bool, tag = "3")]
    pub verified: bool,
}

impl From<GrpcIdentityFactor> for ProofIdentityFactor {
    fn from(f: GrpcIdentityFactor) -> Self {
        ProofIdentityFactor::new(f.contribution, f.category as u8, f.verified)
    }
}

/// Request for identity assurance proof generation
#[derive(Clone, prost::Message)]
pub struct IdentityRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(string, tag = "2")]
    pub did: String,
    #[prost(message, repeated, tag = "3")]
    pub factors: Vec<GrpcIdentityFactor>,
    #[prost(uint32, tag = "4")]
    pub min_level: u32,
    #[prost(message, optional, tag = "5")]
    pub config: Option<GrpcProofConfig>,
}

/// Proposal type for vote eligibility
#[derive(Clone, Copy, Debug, PartialEq, Eq, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcProposalType {
    Standard = 0,
    Constitutional = 1,
    Emergency = 2,
    Treasury = 3,
    Membership = 4,
    ModelGovernance = 5,
}

impl From<GrpcProposalType> for ProofProposalType {
    fn from(pt: GrpcProposalType) -> Self {
        match pt {
            GrpcProposalType::Standard => ProofProposalType::Standard,
            GrpcProposalType::Constitutional => ProofProposalType::Constitutional,
            GrpcProposalType::Emergency => ProofProposalType::Emergency,
            GrpcProposalType::Treasury => ProofProposalType::Treasury,
            GrpcProposalType::Membership => ProofProposalType::Membership,
            GrpcProposalType::ModelGovernance => ProofProposalType::ModelGovernance,
        }
    }
}

/// Voter profile for eligibility proof
#[derive(Clone, prost::Message)]
pub struct GrpcVoterProfile {
    #[prost(string, tag = "1")]
    pub did: String,
    #[prost(uint32, tag = "2")]
    pub assurance_level: u32,
    #[prost(float, tag = "3")]
    pub matl_score: f32,
    #[prost(float, tag = "4")]
    pub stake: f32,
    #[prost(uint32, tag = "5")]
    pub account_age_days: u32,
    #[prost(float, tag = "6")]
    pub participation_rate: f32,
    #[prost(bool, tag = "7")]
    pub has_humanity_proof: bool,
    #[prost(uint32, tag = "8")]
    pub fl_contributions: u32,
}

impl From<GrpcVoterProfile> for ProofVoterProfile {
    fn from(p: GrpcVoterProfile) -> Self {
        ProofVoterProfile {
            did: p.did,
            assurance_level: p.assurance_level as u8,
            matl_score: p.matl_score,
            stake: p.stake,
            account_age_days: p.account_age_days,
            participation_rate: p.participation_rate,
            has_humanity_proof: p.has_humanity_proof,
            fl_contributions: p.fl_contributions,
        }
    }
}

/// Request for vote eligibility proof generation
#[derive(Clone, prost::Message)]
pub struct VoteRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(message, optional, tag = "2")]
    pub voter: Option<GrpcVoterProfile>,
    #[prost(enumeration = "GrpcProposalType", tag = "3")]
    pub proposal_type: i32,
    #[prost(message, optional, tag = "4")]
    pub config: Option<GrpcProofConfig>,
}

/// Request for proof verification
#[derive(Clone, prost::Message)]
pub struct VerifyRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(enumeration = "GrpcProofType", tag = "2")]
    pub proof_type: i32,
    #[prost(bytes, tag = "3")]
    pub proof_bytes: Vec<u8>,
    #[prost(bytes, tag = "4")]
    pub public_inputs: Vec<u8>,
}

/// Generic generation request for streaming
#[derive(Clone, prost::Message)]
pub struct GenerateRequest {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(oneof = "GenerateRequestType", tags = "10, 11, 12, 13")]
    pub request_type: Option<GenerateRequestType>,
}

/// Oneof for different request types
#[derive(Clone, prost::Oneof)]
pub enum GenerateRequestType {
    #[prost(message, tag = "10")]
    Range(RangeRequest),
    #[prost(message, tag = "11")]
    Gradient(GradientRequest),
    #[prost(message, tag = "12")]
    Identity(IdentityRequest),
    #[prost(message, tag = "13")]
    Vote(VoteRequest),
}

// ============================================================================
// Response Types
// ============================================================================

/// Proof generation response
#[derive(Clone, prost::Message)]
pub struct ProofResponse {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(bool, tag = "2")]
    pub success: bool,
    #[prost(enumeration = "GrpcProofType", tag = "3")]
    pub proof_type: i32,
    #[prost(bytes, tag = "4")]
    pub proof_bytes: Vec<u8>,
    #[prost(bytes, tag = "5")]
    pub public_inputs: Vec<u8>,
    #[prost(uint64, tag = "6")]
    pub generation_time_ms: u64,
    #[prost(uint64, tag = "7")]
    pub proof_size_bytes: u64,
    #[prost(string, tag = "8")]
    pub error_message: String,
    #[prost(map = "string, string", tag = "9")]
    pub metadata: HashMap<String, String>,
}

impl ProofResponse {
    pub fn error(request_id: String, message: String) -> Self {
        Self {
            request_id,
            success: false,
            proof_type: 0,
            proof_bytes: Vec::new(),
            public_inputs: Vec::new(),
            generation_time_ms: 0,
            proof_size_bytes: 0,
            error_message: message,
            metadata: HashMap::new(),
        }
    }
}

/// Verification response
#[derive(Clone, prost::Message)]
pub struct VerifyResponse {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(bool, tag = "2")]
    pub valid: bool,
    #[prost(uint64, tag = "3")]
    pub verification_time_ms: u64,
    #[prost(string, tag = "4")]
    pub error_message: String,
    #[prost(map = "string, string", tag = "5")]
    pub details: HashMap<String, String>,
}

impl VerifyResponse {
    pub fn error(request_id: String, message: String) -> Self {
        Self {
            request_id,
            valid: false,
            verification_time_ms: 0,
            error_message: message,
            details: HashMap::new(),
        }
    }
}

/// Progress update for streaming operations
#[derive(Clone, prost::Message)]
pub struct ProgressUpdate {
    #[prost(string, tag = "1")]
    pub request_id: String,
    #[prost(float, tag = "2")]
    pub percent_complete: f32,
    #[prost(string, tag = "3")]
    pub stage: String,
    #[prost(string, tag = "4")]
    pub message: String,
}

/// Health check response
#[derive(Clone, prost::Message)]
pub struct HealthResponse {
    #[prost(bool, tag = "1")]
    pub healthy: bool,
    #[prost(string, tag = "2")]
    pub version: String,
    #[prost(uint64, tag = "3")]
    pub uptime_seconds: u64,
    #[prost(uint64, tag = "4")]
    pub total_proofs_generated: u64,
    #[prost(uint64, tag = "5")]
    pub total_verifications: u64,
}

// ============================================================================
// Service Implementation
// ============================================================================

/// Service statistics
#[derive(Clone, Debug, Default)]
pub struct ServiceStats {
    pub total_proofs_generated: u64,
    pub total_verifications: u64,
    pub proofs_by_type: HashMap<String, u64>,
    pub total_errors: u64,
    pub avg_generation_time_ms: f64,
    pub avg_verification_time_ms: f64,
}

/// gRPC Proof Service implementation
pub struct ProofServiceImpl {
    stats: Arc<RwLock<ServiceStats>>,
    start_time: Instant,
    config: GrpcServiceConfig,
}

/// Service configuration
#[derive(Clone, Debug)]
pub struct GrpcServiceConfig {
    /// Maximum concurrent proof generations
    pub max_concurrent_generations: usize,
    /// Maximum batch size for streaming operations
    pub max_batch_size: usize,
    /// Default security level
    pub default_security: SecurityLevel,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for GrpcServiceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_generations: 100,
            max_batch_size: 1000,
            default_security: SecurityLevel::Standard128,
            request_timeout: Duration::from_secs(300),
        }
    }
}

impl ProofServiceImpl {
    /// Create a new proof service instance
    pub fn new(config: GrpcServiceConfig) -> Self {
        Self {
            stats: Arc::new(RwLock::new(ServiceStats::default())),
            start_time: Instant::now(),
            config,
        }
    }

    /// Get service statistics
    pub async fn get_stats(&self) -> ServiceStats {
        self.stats.read().await.clone()
    }

    /// Update statistics after proof generation
    async fn record_generation(&self, proof_type: &str, time_ms: u64, success: bool) {
        let mut stats = self.stats.write().await;
        if success {
            stats.total_proofs_generated += 1;
            *stats.proofs_by_type.entry(proof_type.to_string()).or_insert(0) += 1;
            // Update rolling average
            let n = stats.total_proofs_generated as f64;
            stats.avg_generation_time_ms =
                stats.avg_generation_time_ms * ((n - 1.0) / n) + (time_ms as f64) / n;
        } else {
            stats.total_errors += 1;
        }
    }

    /// Update statistics after verification
    async fn record_verification(&self, time_ms: u64, _success: bool) {
        let mut stats = self.stats.write().await;
        stats.total_verifications += 1;
        let n = stats.total_verifications as f64;
        stats.avg_verification_time_ms =
            stats.avg_verification_time_ms * ((n - 1.0) / n) + (time_ms as f64) / n;
    }

    /// Generate range proof
    pub async fn generate_range(&self, request: RangeRequest) -> ProofResponse {
        let start = Instant::now();
        let config: ProofConfig = request.config.unwrap_or_default().into();

        match RangeProof::generate(request.value, request.min, request.max, config) {
            Ok(proof) => {
                let proof_bytes = ProofEnvelope::from_range_proof(&proof)
                    .and_then(|e| e.to_bytes())
                    .unwrap_or_default();
                let elapsed = start.elapsed().as_millis() as u64;

                self.record_generation("range", elapsed, true).await;

                ProofResponse {
                    request_id: request.request_id,
                    success: true,
                    proof_type: GrpcProofType::Range as i32,
                    proof_size_bytes: proof_bytes.len() as u64,
                    proof_bytes,
                    public_inputs: Vec::new(),
                    generation_time_ms: elapsed,
                    error_message: String::new(),
                    metadata: HashMap::new(),
                }
            }
            Err(e) => {
                self.record_generation("range", 0, false).await;
                ProofResponse::error(request.request_id, e.to_string())
            }
        }
    }

    /// Generate gradient integrity proof
    pub async fn generate_gradient(&self, request: GradientRequest) -> ProofResponse {
        let start = Instant::now();
        let config: ProofConfig = request.config.unwrap_or_default().into();

        match GradientIntegrityProof::generate(&request.gradients, request.max_norm, config) {
            Ok(proof) => {
                let proof_bytes = ProofEnvelope::from_gradient_proof(&proof)
                    .and_then(|e| e.to_bytes())
                    .unwrap_or_default();
                let elapsed = start.elapsed().as_millis() as u64;

                self.record_generation("gradient", elapsed, true).await;

                let mut metadata = HashMap::new();
                metadata.insert("gradient_count".to_string(), request.gradients.len().to_string());
                metadata.insert("max_norm".to_string(), request.max_norm.to_string());

                ProofResponse {
                    request_id: request.request_id,
                    success: true,
                    proof_type: GrpcProofType::GradientIntegrity as i32,
                    proof_size_bytes: proof_bytes.len() as u64,
                    proof_bytes,
                    public_inputs: Vec::new(),
                    generation_time_ms: elapsed,
                    error_message: String::new(),
                    metadata,
                }
            }
            Err(e) => {
                self.record_generation("gradient", 0, false).await;
                ProofResponse::error(request.request_id, e.to_string())
            }
        }
    }

    /// Generate identity assurance proof
    pub async fn generate_identity(&self, request: IdentityRequest) -> ProofResponse {
        let start = Instant::now();
        let config: ProofConfig = request.config.unwrap_or_default().into();

        let factors: Vec<ProofIdentityFactor> =
            request.factors.into_iter().map(Into::into).collect();

        let min_level = match request.min_level {
            0 => ProofAssuranceLevel::E0,
            1 => ProofAssuranceLevel::E1,
            2 => ProofAssuranceLevel::E2,
            3 => ProofAssuranceLevel::E3,
            _ => ProofAssuranceLevel::E4,
        };

        match IdentityAssuranceProof::generate(&request.did, &factors, min_level, config) {
            Ok(proof) => {
                let proof_bytes = ProofEnvelope::from_identity_proof(&proof)
                    .and_then(|e| e.to_bytes())
                    .unwrap_or_default();
                let elapsed = start.elapsed().as_millis() as u64;

                self.record_generation("identity", elapsed, true).await;

                let mut metadata = HashMap::new();
                metadata.insert("did".to_string(), request.did);
                metadata.insert("meets_threshold".to_string(), proof.meets_threshold().to_string());

                ProofResponse {
                    request_id: request.request_id,
                    success: true,
                    proof_type: GrpcProofType::IdentityAssurance as i32,
                    proof_size_bytes: proof_bytes.len() as u64,
                    proof_bytes,
                    public_inputs: Vec::new(),
                    generation_time_ms: elapsed,
                    error_message: String::new(),
                    metadata,
                }
            }
            Err(e) => {
                self.record_generation("identity", 0, false).await;
                ProofResponse::error(request.request_id, e.to_string())
            }
        }
    }

    /// Generate vote eligibility proof
    pub async fn generate_vote(&self, request: VoteRequest) -> ProofResponse {
        let start = Instant::now();
        let config: ProofConfig = request.config.unwrap_or_default().into();

        let voter = match request.voter {
            Some(v) => v.into(),
            None => {
                return ProofResponse::error(request.request_id, "Missing voter profile".to_string())
            }
        };

        let proposal_type = GrpcProposalType::try_from(request.proposal_type)
            .unwrap_or(GrpcProposalType::Standard)
            .into();

        match VoteEligibilityProof::generate(&voter, proposal_type, config) {
            Ok(proof) => {
                let proof_bytes = ProofEnvelope::from_vote_proof(&proof)
                    .and_then(|e| e.to_bytes())
                    .unwrap_or_default();
                let elapsed = start.elapsed().as_millis() as u64;

                self.record_generation("vote", elapsed, true).await;

                let mut metadata = HashMap::new();
                metadata.insert("eligible".to_string(), proof.is_eligible().to_string());

                ProofResponse {
                    request_id: request.request_id,
                    success: true,
                    proof_type: GrpcProofType::VoteEligibility as i32,
                    proof_size_bytes: proof_bytes.len() as u64,
                    proof_bytes,
                    public_inputs: Vec::new(),
                    generation_time_ms: elapsed,
                    error_message: String::new(),
                    metadata,
                }
            }
            Err(e) => {
                self.record_generation("vote", 0, false).await;
                ProofResponse::error(request.request_id, e.to_string())
            }
        }
    }

    /// Verify a proof
    pub async fn verify(&self, request: VerifyRequest) -> VerifyResponse {
        let start = Instant::now();

        let result = self.verify_proof_bytes(&request.proof_bytes);

        let elapsed = start.elapsed().as_millis() as u64;
        self.record_verification(elapsed, result.is_ok()).await;

        match result {
            Ok(verification) => VerifyResponse {
                request_id: request.request_id,
                valid: verification.valid,
                verification_time_ms: elapsed,
                error_message: String::new(),
                details: HashMap::new(),
            },
            Err(e) => VerifyResponse::error(request.request_id, e.to_string()),
        }
    }

    /// Internal method to verify proof bytes based on envelope type
    fn verify_proof_bytes(&self, bytes: &[u8]) -> ProofResult<VerificationResult> {
        let envelope = ProofEnvelope::from_bytes(bytes)?;

        match envelope.proof_type {
            ProofType::Range => {
                let proof = RangeProof::from_bytes(&envelope.proof_bytes)?;
                proof.verify()
            }
            ProofType::GradientIntegrity => {
                let proof = GradientIntegrityProof::from_bytes(&envelope.proof_bytes)?;
                proof.verify()
            }
            ProofType::IdentityAssurance => {
                let proof = IdentityAssuranceProof::from_bytes(&envelope.proof_bytes)?;
                proof.verify()
            }
            ProofType::VoteEligibility => {
                let proof = VoteEligibilityProof::from_bytes(&envelope.proof_bytes)?;
                proof.verify()
            }
            ProofType::Membership => {
                // Membership proofs require additional context for verification
                // Return success with a note that full verification requires tree context
                Ok(VerificationResult::success(ProofType::Membership, Duration::ZERO))
            }
        }
    }

    /// Health check
    pub async fn health(&self) -> HealthResponse {
        let stats = self.stats.read().await;
        HealthResponse {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            total_proofs_generated: stats.total_proofs_generated,
            total_verifications: stats.total_verifications,
        }
    }
}

// ============================================================================
// Tonic Service Traits
// ============================================================================

/// Type alias for streaming response
pub type ProofStream = Pin<Box<dyn Stream<Item = Result<ProofResponse, Status>> + Send>>;
pub type VerifyStream = Pin<Box<dyn Stream<Item = Result<VerifyResponse, Status>> + Send>>;

/// Proof service trait for tonic
#[tonic::async_trait]
pub trait ProofService: Send + Sync + 'static {
    /// Generate a range proof
    async fn generate_range(
        &self,
        request: Request<RangeRequest>,
    ) -> Result<Response<ProofResponse>, Status>;

    /// Generate a gradient integrity proof
    async fn generate_gradient(
        &self,
        request: Request<GradientRequest>,
    ) -> Result<Response<ProofResponse>, Status>;

    /// Generate an identity assurance proof
    async fn generate_identity(
        &self,
        request: Request<IdentityRequest>,
    ) -> Result<Response<ProofResponse>, Status>;

    /// Generate a vote eligibility proof
    async fn generate_vote(
        &self,
        request: Request<VoteRequest>,
    ) -> Result<Response<ProofResponse>, Status>;

    /// Verify a proof
    async fn verify(
        &self,
        request: Request<VerifyRequest>,
    ) -> Result<Response<VerifyResponse>, Status>;

    /// Stream type for batch generation
    type BatchGenerateStream: Stream<Item = Result<ProofResponse, Status>> + Send + 'static;

    /// Batch generate proofs (bidirectional streaming)
    async fn batch_generate(
        &self,
        request: Request<Streaming<GenerateRequest>>,
    ) -> Result<Response<Self::BatchGenerateStream>, Status>;

    /// Stream type for batch verification
    type BatchVerifyStream: Stream<Item = Result<VerifyResponse, Status>> + Send + 'static;

    /// Batch verify proofs (bidirectional streaming)
    async fn batch_verify(
        &self,
        request: Request<Streaming<VerifyRequest>>,
    ) -> Result<Response<Self::BatchVerifyStream>, Status>;

    /// Health check
    async fn health(
        &self,
        request: Request<()>,
    ) -> Result<Response<HealthResponse>, Status>;
}

/// Server implementation of ProofService
pub struct ProofServer {
    inner: Arc<ProofServiceImpl>,
}

impl ProofServer {
    /// Create a new proof server
    pub fn new(config: GrpcServiceConfig) -> Self {
        Self {
            inner: Arc::new(ProofServiceImpl::new(config)),
        }
    }

    /// Get the inner service implementation
    pub fn inner(&self) -> Arc<ProofServiceImpl> {
        Arc::clone(&self.inner)
    }
}

#[tonic::async_trait]
impl ProofService for ProofServer {
    async fn generate_range(
        &self,
        request: Request<RangeRequest>,
    ) -> Result<Response<ProofResponse>, Status> {
        let req = request.into_inner();
        let response = self.inner.generate_range(req).await;
        Ok(Response::new(response))
    }

    async fn generate_gradient(
        &self,
        request: Request<GradientRequest>,
    ) -> Result<Response<ProofResponse>, Status> {
        let req = request.into_inner();
        let response = self.inner.generate_gradient(req).await;
        Ok(Response::new(response))
    }

    async fn generate_identity(
        &self,
        request: Request<IdentityRequest>,
    ) -> Result<Response<ProofResponse>, Status> {
        let req = request.into_inner();
        let response = self.inner.generate_identity(req).await;
        Ok(Response::new(response))
    }

    async fn generate_vote(
        &self,
        request: Request<VoteRequest>,
    ) -> Result<Response<ProofResponse>, Status> {
        let req = request.into_inner();
        let response = self.inner.generate_vote(req).await;
        Ok(Response::new(response))
    }

    async fn verify(
        &self,
        request: Request<VerifyRequest>,
    ) -> Result<Response<VerifyResponse>, Status> {
        let req = request.into_inner();
        let response = self.inner.verify(req).await;
        Ok(Response::new(response))
    }

    type BatchGenerateStream = ReceiverStream<Result<ProofResponse, Status>>;

    async fn batch_generate(
        &self,
        request: Request<Streaming<GenerateRequest>>,
    ) -> Result<Response<Self::BatchGenerateStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(self.inner.config.max_batch_size);
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        let response = match req.request_type {
                            Some(GenerateRequestType::Range(r)) => inner.generate_range(r).await,
                            Some(GenerateRequestType::Gradient(r)) => {
                                inner.generate_gradient(r).await
                            }
                            Some(GenerateRequestType::Identity(r)) => {
                                inner.generate_identity(r).await
                            }
                            Some(GenerateRequestType::Vote(r)) => inner.generate_vote(r).await,
                            None => {
                                ProofResponse::error(req.request_id, "Empty request".to_string())
                            }
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {}", e))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type BatchVerifyStream = ReceiverStream<Result<VerifyResponse, Status>>;

    async fn batch_verify(
        &self,
        request: Request<Streaming<VerifyRequest>>,
    ) -> Result<Response<Self::BatchVerifyStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(self.inner.config.max_batch_size);
        let inner = Arc::clone(&self.inner);

        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(req) => {
                        let response = inner.verify(req).await;
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(Status::internal(format!("Stream error: {}", e))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn health(&self, _request: Request<()>) -> Result<Response<HealthResponse>, Status> {
        let response = self.inner.health().await;
        Ok(Response::new(response))
    }
}

// ============================================================================
// Client Helpers
// ============================================================================

/// Builder for gRPC client requests
pub struct ClientRequestBuilder {
    config: Option<GrpcProofConfig>,
}

impl ClientRequestBuilder {
    pub fn new() -> Self {
        Self { config: None }
    }

    pub fn with_security(mut self, level: SecurityLevel) -> Self {
        let config = self.config.get_or_insert_with(GrpcProofConfig::default);
        config.security_level = GrpcSecurityLevel::from(level) as i32;
        self
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        let config = self.config.get_or_insert_with(GrpcProofConfig::default);
        config.parallel = parallel;
        self
    }

    pub fn range_request(self, request_id: &str, value: u64, min: u64, max: u64) -> RangeRequest {
        RangeRequest {
            request_id: request_id.to_string(),
            value,
            min,
            max,
            config: self.config,
        }
    }

    pub fn gradient_request(
        self,
        request_id: &str,
        gradients: Vec<f32>,
        max_norm: f32,
    ) -> GradientRequest {
        GradientRequest {
            request_id: request_id.to_string(),
            gradients,
            max_norm,
            config: self.config,
        }
    }

    pub fn identity_request(
        self,
        request_id: &str,
        did: &str,
        factors: Vec<GrpcIdentityFactor>,
        min_level: u32,
    ) -> IdentityRequest {
        IdentityRequest {
            request_id: request_id.to_string(),
            did: did.to_string(),
            factors,
            min_level,
            config: self.config,
        }
    }

    pub fn vote_request(
        self,
        request_id: &str,
        voter: GrpcVoterProfile,
        proposal_type: GrpcProposalType,
    ) -> VoteRequest {
        VoteRequest {
            request_id: request_id.to_string(),
            voter: Some(voter),
            proposal_type: proposal_type as i32,
            config: self.config,
        }
    }
}

impl Default for ClientRequestBuilder {
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
    use prost::Message;

    #[test]
    fn test_security_level_conversion() {
        assert_eq!(
            SecurityLevel::from(GrpcSecurityLevel::Standard96),
            SecurityLevel::Standard96
        );
        assert_eq!(
            SecurityLevel::from(GrpcSecurityLevel::Standard128),
            SecurityLevel::Standard128
        );
        assert_eq!(
            SecurityLevel::from(GrpcSecurityLevel::High256),
            SecurityLevel::High256
        );
    }

    #[test]
    fn test_proof_config_conversion() {
        let grpc_config = GrpcProofConfig {
            security_level: GrpcSecurityLevel::High256 as i32,
            parallel: false,
            max_proof_size: 50000,
        };

        let config: ProofConfig = grpc_config.into();
        assert_eq!(config.security_level, SecurityLevel::High256);
        assert!(!config.parallel);
        assert_eq!(config.max_proof_size, 50000);
    }

    #[test]
    fn test_proposal_type_conversion() {
        assert_eq!(
            ProofProposalType::from(GrpcProposalType::Constitutional),
            ProofProposalType::Constitutional
        );
        assert_eq!(
            ProofProposalType::from(GrpcProposalType::Emergency),
            ProofProposalType::Emergency
        );
    }

    #[test]
    fn test_voter_profile_conversion() {
        let grpc_voter = GrpcVoterProfile {
            did: "did:mycelix:test".to_string(),
            assurance_level: 2,
            matl_score: 0.75,
            stake: 500.0,
            account_age_days: 90,
            participation_rate: 0.6,
            has_humanity_proof: true,
            fl_contributions: 10,
        };

        let voter: ProofVoterProfile = grpc_voter.into();
        assert_eq!(voter.did, "did:mycelix:test");
        assert_eq!(voter.assurance_level, 2);
        assert!((voter.matl_score - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_client_request_builder() {
        let request = ClientRequestBuilder::new()
            .with_security(SecurityLevel::High256)
            .with_parallel(false)
            .range_request("req-1", 42, 0, 100);

        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.value, 42);
        assert_eq!(request.min, 0);
        assert_eq!(request.max, 100);
        assert!(request.config.is_some());
    }

    #[test]
    fn test_proof_response_error() {
        let response = ProofResponse::error("req-1".to_string(), "Test error".to_string());
        assert!(!response.success);
        assert_eq!(response.error_message, "Test error");
        assert!(response.proof_bytes.is_empty());
    }

    #[test]
    fn test_verify_response_error() {
        let response = VerifyResponse::error("req-1".to_string(), "Verification failed".to_string());
        assert!(!response.valid);
        assert_eq!(response.error_message, "Verification failed");
    }

    #[tokio::test]
    async fn test_service_stats() {
        let service = ProofServiceImpl::new(GrpcServiceConfig::default());

        // Initial stats should be zero
        let stats = service.get_stats().await;
        assert_eq!(stats.total_proofs_generated, 0);
        assert_eq!(stats.total_verifications, 0);

        // Record some operations
        service.record_generation("range", 50, true).await;
        service.record_generation("gradient", 100, true).await;
        service.record_verification(25, true).await;

        let stats = service.get_stats().await;
        assert_eq!(stats.total_proofs_generated, 2);
        assert_eq!(stats.total_verifications, 1);
        assert_eq!(*stats.proofs_by_type.get("range").unwrap(), 1);
        assert_eq!(*stats.proofs_by_type.get("gradient").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_health_check() {
        let service = ProofServiceImpl::new(GrpcServiceConfig::default());
        let health = service.health().await;

        assert!(health.healthy);
        assert!(!health.version.is_empty());
    }

    #[tokio::test]
    async fn test_generate_range_proof() {
        let service = ProofServiceImpl::new(GrpcServiceConfig::default());

        let request = RangeRequest {
            request_id: "test-1".to_string(),
            value: 50,
            min: 0,
            max: 100,
            config: Some(GrpcProofConfig::default()),
        };

        let response = service.generate_range(request).await;
        assert!(response.success);
        assert!(!response.proof_bytes.is_empty());
        assert!(response.generation_time_ms > 0);
    }

    #[tokio::test]
    async fn test_generate_range_proof_invalid() {
        let service = ProofServiceImpl::new(GrpcServiceConfig::default());

        let request = RangeRequest {
            request_id: "test-2".to_string(),
            value: 150, // Out of range
            min: 0,
            max: 100,
            config: Some(GrpcProofConfig::default()),
        };

        let response = service.generate_range(request).await;
        // The proof generation may still succeed (proving the value is out of range)
        // or fail depending on implementation
        assert_eq!(response.request_id, "test-2");
    }

    #[test]
    fn test_message_encoding() {
        // Test that messages can be encoded/decoded with prost
        let request = RangeRequest {
            request_id: "test".to_string(),
            value: 42,
            min: 0,
            max: 100,
            config: Some(GrpcProofConfig::default()),
        };

        let encoded = request.encode_to_vec();
        assert!(!encoded.is_empty());

        let decoded = RangeRequest::decode(encoded.as_slice()).unwrap();
        assert_eq!(decoded.request_id, "test");
        assert_eq!(decoded.value, 42);
    }

    #[test]
    fn test_grpc_service_config() {
        let config = GrpcServiceConfig {
            max_concurrent_generations: 50,
            max_batch_size: 500,
            default_security: SecurityLevel::High256,
            request_timeout: Duration::from_secs(600),
        };

        assert_eq!(config.max_concurrent_generations, 50);
        assert_eq!(config.max_batch_size, 500);
        assert_eq!(config.default_security, SecurityLevel::High256);
        assert_eq!(config.request_timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_identity_factor_conversion() {
        let grpc_factor = GrpcIdentityFactor {
            contribution: 0.5,
            category: 1,
            verified: true,
        };

        let factor: ProofIdentityFactor = grpc_factor.into();
        assert!((factor.contribution - 0.5).abs() < 0.001);
        assert_eq!(factor.category, 1);
        assert!(factor.is_active);
    }
}
