//! Multi-Party Computation (MPC) Proof Generation
//!
//! Enables collaborative proof generation across multiple parties:
//! - Secret sharing of witness
//! - Distributed computation
//! - Threshold signatures for proofs
//! - Privacy-preserving aggregation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};

use crate::proofs::{ProofConfig, ProofType};

// ============================================================================
// MPC Configuration
// ============================================================================

/// MPC protocol configuration
#[derive(Clone, Debug)]
pub struct MpcConfig {
    /// Minimum number of parties required
    pub threshold: usize,
    /// Total number of parties
    pub total_parties: usize,
    /// Communication timeout
    pub timeout: Duration,
    /// Number of rounds in the protocol
    pub num_rounds: usize,
    /// Enable preprocessing
    pub preprocessing: bool,
    /// Secret sharing scheme
    pub sharing_scheme: SharingScheme,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            threshold: 2,
            total_parties: 3,
            timeout: Duration::from_secs(60),
            num_rounds: 3,
            preprocessing: true,
            sharing_scheme: SharingScheme::Shamir,
        }
    }
}

/// Secret sharing scheme
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharingScheme {
    /// Shamir's Secret Sharing
    Shamir,
    /// Additive sharing
    Additive,
    /// Replicated sharing
    Replicated,
}

// ============================================================================
// Party Management
// ============================================================================

/// MPC party identifier
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartyId(pub String);

impl PartyId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Party information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyInfo {
    /// Party identifier
    pub id: PartyId,
    /// Party index (1-based)
    pub index: usize,
    /// Public key for verification
    pub public_key: Vec<u8>,
    /// Network endpoint
    pub endpoint: Option<String>,
    /// Party status
    pub status: PartyStatus,
}

/// Party status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyStatus {
    Pending,
    Ready,
    Computing,
    Done,
    Failed,
    Timeout,
}

// ============================================================================
// MPC Session
// ============================================================================

/// MPC session for proof generation
#[allow(dead_code)]
pub struct MpcSession {
    /// Session identifier
    pub id: String,
    /// Configuration
    config: MpcConfig,
    /// Proof type being generated
    proof_type: ProofType,
    /// Proof configuration
    proof_config: ProofConfig,
    /// Participating parties
    parties: Arc<RwLock<HashMap<PartyId, PartyInfo>>>,
    /// Session state
    state: Arc<RwLock<SessionState>>,
    /// Shares received
    shares: Arc<RwLock<HashMap<PartyId, Share>>>,
    /// Partial results
    partial_results: Arc<RwLock<HashMap<PartyId, PartialResult>>>,
    /// Message channels
    message_tx: mpsc::Sender<MpcMessage>,
    message_rx: Arc<RwLock<mpsc::Receiver<MpcMessage>>>,
    /// Start time
    start_time: Instant,
}

/// Session state
#[derive(Clone, Debug)]
pub struct SessionState {
    pub phase: MpcPhase,
    pub current_round: usize,
    pub parties_ready: usize,
    pub shares_received: usize,
    pub results_received: usize,
    pub error: Option<String>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            phase: MpcPhase::Setup,
            current_round: 0,
            parties_ready: 0,
            shares_received: 0,
            results_received: 0,
            error: None,
        }
    }
}

/// MPC protocol phase
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MpcPhase {
    Setup,
    ShareDistribution,
    Computation,
    ResultCollection,
    Finalization,
    Complete,
    Failed,
}

/// A secret share
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Share {
    /// Party that holds this share
    pub holder: PartyId,
    /// Share index
    pub index: usize,
    /// Share value (field element)
    pub value: Vec<u8>,
    /// Commitment to the share
    pub commitment: [u8; 32],
}

/// Partial computation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartialResult {
    /// Party that computed this
    pub party: PartyId,
    /// Round number
    pub round: usize,
    /// Partial trace data
    pub trace_share: Vec<u8>,
    /// Partial constraint evaluation
    pub constraint_share: Vec<u8>,
    /// Verification data
    pub verification: [u8; 32],
}

/// MPC message
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MpcMessage {
    /// Sender party
    pub from: PartyId,
    /// Message type
    pub msg_type: MpcMessageType,
    /// Message payload
    pub payload: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// Message type
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MpcMessageType {
    Ready,
    Share(Share),
    PartialResult(PartialResult),
    Abort(String),
    Complete,
}

impl MpcSession {
    /// Create a new MPC session
    pub fn new(
        proof_type: ProofType,
        proof_config: ProofConfig,
        mpc_config: MpcConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel(100);

        let id = format!(
            "mpc-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros(),
            rand::random::<u16>()
        );

        Self {
            id,
            config: mpc_config,
            proof_type,
            proof_config,
            parties: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(SessionState::default())),
            shares: Arc::new(RwLock::new(HashMap::new())),
            partial_results: Arc::new(RwLock::new(HashMap::new())),
            message_tx: tx,
            message_rx: Arc::new(RwLock::new(rx)),
            start_time: Instant::now(),
        }
    }

    /// Add a party to the session
    pub async fn add_party(&self, info: PartyInfo) -> Result<(), MpcError> {
        let mut parties = self.parties.write().await;

        if parties.len() >= self.config.total_parties {
            return Err(MpcError::TooManyParties);
        }

        parties.insert(info.id.clone(), info);
        Ok(())
    }

    /// Check if session has enough parties
    pub async fn has_quorum(&self) -> bool {
        let parties = self.parties.read().await;
        parties.len() >= self.config.threshold
    }

    /// Start the MPC protocol
    pub async fn start(&self) -> Result<(), MpcError> {
        if !self.has_quorum().await {
            return Err(MpcError::InsufficientParties);
        }

        let mut state = self.state.write().await;
        state.phase = MpcPhase::ShareDistribution;
        state.current_round = 1;

        Ok(())
    }

    /// Receive a message
    pub async fn receive_message(&self, msg: MpcMessage) -> Result<(), MpcError> {
        // Verify sender is a valid party
        let parties = self.parties.read().await;
        if !parties.contains_key(&msg.from) {
            return Err(MpcError::UnknownParty(msg.from.0.clone()));
        }
        drop(parties);

        match msg.msg_type {
            MpcMessageType::Ready => {
                let mut state = self.state.write().await;
                state.parties_ready += 1;
            }
            MpcMessageType::Share(share) => {
                let mut shares = self.shares.write().await;
                shares.insert(msg.from, share);

                let mut state = self.state.write().await;
                state.shares_received += 1;

                // Check if we have all shares
                if state.shares_received >= self.config.threshold {
                    state.phase = MpcPhase::Computation;
                }
            }
            MpcMessageType::PartialResult(result) => {
                let mut results = self.partial_results.write().await;
                results.insert(msg.from, result);

                let mut state = self.state.write().await;
                state.results_received += 1;

                // Check if we have all results
                if state.results_received >= self.config.threshold {
                    state.phase = MpcPhase::ResultCollection;
                }
            }
            MpcMessageType::Abort(reason) => {
                let mut state = self.state.write().await;
                state.phase = MpcPhase::Failed;
                state.error = Some(reason);
            }
            MpcMessageType::Complete => {
                // Party signals completion
            }
        }

        Ok(())
    }

    /// Get current session state
    pub async fn get_state(&self) -> SessionState {
        self.state.read().await.clone()
    }

    /// Generate shares for a witness value
    pub fn generate_shares(&self, witness: &[u8]) -> Vec<Share> {
        match self.config.sharing_scheme {
            SharingScheme::Shamir => self.shamir_share(witness),
            SharingScheme::Additive => self.additive_share(witness),
            SharingScheme::Replicated => self.replicated_share(witness),
        }
    }

    /// Shamir secret sharing
    fn shamir_share(&self, secret: &[u8]) -> Vec<Share> {
        let n = self.config.total_parties;
        let t = self.config.threshold;

        // Generate random polynomial coefficients
        let mut coeffs: Vec<Vec<u8>> = Vec::with_capacity(t);
        coeffs.push(secret.to_vec());

        for _ in 1..t {
            let random_coeff: Vec<u8> = (0..secret.len()).map(|_| rand::random()).collect();
            coeffs.push(random_coeff);
        }

        // Evaluate polynomial at each point
        (1..=n)
            .map(|i| {
                let mut value = vec![0u8; secret.len()];

                // Evaluate polynomial: f(i) = a_0 + a_1*i + a_2*i^2 + ...
                let mut power = 1u64;
                for coeff in &coeffs {
                    for (j, v) in value.iter_mut().enumerate() {
                        *v = v.wrapping_add((coeff[j] as u64 * power % 256) as u8);
                    }
                    power = power.wrapping_mul(i as u64);
                }

                let commitment = blake3::hash(&value);

                Share {
                    holder: PartyId::new(&format!("party-{}", i)),
                    index: i,
                    value,
                    commitment: *commitment.as_bytes(),
                }
            })
            .collect()
    }

    /// Additive secret sharing
    fn additive_share(&self, secret: &[u8]) -> Vec<Share> {
        let n = self.config.total_parties;

        // Generate n-1 random shares
        let mut shares: Vec<Vec<u8>> = Vec::with_capacity(n);
        for _ in 0..n - 1 {
            let random_share: Vec<u8> = (0..secret.len()).map(|_| rand::random()).collect();
            shares.push(random_share);
        }

        // Compute last share so they sum to secret
        let mut last_share = secret.to_vec();
        for share in &shares {
            for (i, v) in last_share.iter_mut().enumerate() {
                *v = v.wrapping_sub(share[i]);
            }
        }
        shares.push(last_share);

        shares
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let commitment = blake3::hash(&value);
                Share {
                    holder: PartyId::new(&format!("party-{}", i + 1)),
                    index: i + 1,
                    value,
                    commitment: *commitment.as_bytes(),
                }
            })
            .collect()
    }

    /// Replicated secret sharing
    fn replicated_share(&self, secret: &[u8]) -> Vec<Share> {
        // Each party gets a subset of additive shares
        let additive = self.additive_share(secret);

        // In replicated sharing, each party i gets shares {s_j : j != i}
        // Simplified: just return additive shares
        additive
    }

    /// Reconstruct secret from shares
    pub fn reconstruct(&self, shares: &[Share]) -> Result<Vec<u8>, MpcError> {
        if shares.len() < self.config.threshold {
            return Err(MpcError::InsufficientShares);
        }

        match self.config.sharing_scheme {
            SharingScheme::Shamir => self.shamir_reconstruct(shares),
            SharingScheme::Additive => self.additive_reconstruct(shares),
            SharingScheme::Replicated => self.additive_reconstruct(shares),
        }
    }

    /// Shamir reconstruction
    fn shamir_reconstruct(&self, shares: &[Share]) -> Result<Vec<u8>, MpcError> {
        let t = shares.len();
        if t == 0 {
            return Err(MpcError::InsufficientShares);
        }

        let value_len = shares[0].value.len();
        let mut result = vec![0u8; value_len];

        // Lagrange interpolation at x=0
        for i in 0..t {
            let xi = shares[i].index as i64;

            // Compute Lagrange coefficient
            let mut numerator: i64 = 1;
            let mut denominator: i64 = 1;

            for j in 0..t {
                if i != j {
                    let xj = shares[j].index as i64;
                    numerator *= -xj;
                    denominator *= xi - xj;
                }
            }

            // Apply coefficient to share
            let coeff = numerator / denominator;
            for (k, v) in result.iter_mut().enumerate() {
                let contribution = (shares[i].value[k] as i64 * coeff) % 256;
                *v = ((*v as i64 + contribution + 256) % 256) as u8;
            }
        }

        Ok(result)
    }

    /// Additive reconstruction
    fn additive_reconstruct(&self, shares: &[Share]) -> Result<Vec<u8>, MpcError> {
        if shares.is_empty() {
            return Err(MpcError::InsufficientShares);
        }

        let value_len = shares[0].value.len();
        let mut result = vec![0u8; value_len];

        for share in shares {
            for (i, v) in result.iter_mut().enumerate() {
                *v = v.wrapping_add(share.value[i]);
            }
        }

        Ok(result)
    }

    /// Finalize and produce proof
    pub async fn finalize(&self) -> Result<MpcProofResult, MpcError> {
        let state = self.state.read().await;

        if state.phase == MpcPhase::Failed {
            return Err(MpcError::ProtocolFailed(
                state.error.clone().unwrap_or_default(),
            ));
        }

        let results = self.partial_results.read().await;
        if results.len() < self.config.threshold {
            return Err(MpcError::InsufficientResults);
        }

        // Combine partial results
        let mut combined_trace = Vec::new();
        let mut combined_constraint = Vec::new();

        for result in results.values() {
            if combined_trace.is_empty() {
                combined_trace = result.trace_share.clone();
                combined_constraint = result.constraint_share.clone();
            } else {
                // XOR combine (simplified)
                for (i, v) in combined_trace.iter_mut().enumerate() {
                    if i < result.trace_share.len() {
                        *v ^= result.trace_share[i];
                    }
                }
            }
        }

        // Generate final proof hash
        let mut hasher = blake3::Hasher::new();
        hasher.update(&combined_trace);
        hasher.update(&combined_constraint);
        let proof_hash = hasher.finalize();

        Ok(MpcProofResult {
            session_id: self.id.clone(),
            proof_type: self.proof_type,
            proof_bytes: combined_trace,
            proof_hash: proof_hash.as_bytes().to_vec(),
            parties_contributed: results.len(),
            generation_time: self.start_time.elapsed(),
        })
    }
}

/// MPC proof generation result
#[derive(Clone, Debug)]
pub struct MpcProofResult {
    /// Session ID
    pub session_id: String,
    /// Proof type
    pub proof_type: ProofType,
    /// Combined proof bytes
    pub proof_bytes: Vec<u8>,
    /// Proof hash
    pub proof_hash: Vec<u8>,
    /// Number of parties that contributed
    pub parties_contributed: usize,
    /// Total generation time
    pub generation_time: Duration,
}

/// MPC error types
#[derive(Clone, Debug, thiserror::Error)]
pub enum MpcError {
    #[error("Not enough parties for threshold")]
    InsufficientParties,
    #[error("Too many parties")]
    TooManyParties,
    #[error("Unknown party: {0}")]
    UnknownParty(String),
    #[error("Not enough shares for reconstruction")]
    InsufficientShares,
    #[error("Not enough partial results")]
    InsufficientResults,
    #[error("Protocol failed: {0}")]
    ProtocolFailed(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
    #[error("Timeout")]
    Timeout,
    #[error("Invalid share")]
    InvalidShare,
}

// ============================================================================
// MPC Coordinator
// ============================================================================

/// Coordinator for MPC sessions
pub struct MpcCoordinator {
    config: MpcConfig,
    sessions: Arc<RwLock<HashMap<String, Arc<MpcSession>>>>,
    stats: Arc<RwLock<MpcStats>>,
}

/// MPC statistics
#[derive(Clone, Debug, Default)]
pub struct MpcStats {
    pub sessions_created: u64,
    pub sessions_completed: u64,
    pub sessions_failed: u64,
    pub total_parties_participated: u64,
    pub avg_session_time_ms: f64,
}

impl MpcCoordinator {
    /// Create a new coordinator
    pub fn new(config: MpcConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(MpcStats::default())),
        }
    }

    /// Create a new session
    pub async fn create_session(
        &self,
        proof_type: ProofType,
        proof_config: ProofConfig,
    ) -> Arc<MpcSession> {
        let session = MpcSession::new(proof_type, proof_config, self.config.clone());
        let session_id = session.id.clone();
        let session_arc = Arc::new(session);

        self.sessions
            .write()
            .await
            .insert(session_id, Arc::clone(&session_arc));

        let mut stats = self.stats.write().await;
        stats.sessions_created += 1;

        session_arc
    }

    /// Get a session by ID
    pub async fn get_session(&self, session_id: &str) -> Option<Arc<MpcSession>> {
        self.sessions.read().await.get(session_id).cloned()
    }

    /// Get statistics
    pub async fn stats(&self) -> MpcStats {
        self.stats.read().await.clone()
    }

    /// Clean up completed sessions
    pub async fn cleanup(&self, max_age: Duration) {
        let mut sessions = self.sessions.write().await;

        sessions.retain(|_, session| {
            session.start_time.elapsed() < max_age
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpc_config_default() {
        let config = MpcConfig::default();
        assert_eq!(config.threshold, 2);
        assert_eq!(config.total_parties, 3);
    }

    #[test]
    fn test_party_id() {
        let id = PartyId::new("test-party");
        assert_eq!(id.0, "test-party");
    }

    #[tokio::test]
    async fn test_create_session() {
        let coordinator = MpcCoordinator::new(MpcConfig::default());

        let session = coordinator
            .create_session(ProofType::Range, ProofConfig::default())
            .await;

        assert!(!session.id.is_empty());
    }

    #[tokio::test]
    async fn test_add_parties() {
        let session = MpcSession::new(
            ProofType::Range,
            ProofConfig::default(),
            MpcConfig::default(),
        );

        for i in 1..=3 {
            let party = PartyInfo {
                id: PartyId::new(&format!("party-{}", i)),
                index: i,
                public_key: vec![0u8; 32],
                endpoint: None,
                status: PartyStatus::Pending,
            };
            session.add_party(party).await.unwrap();
        }

        assert!(session.has_quorum().await);
    }

    #[tokio::test]
    async fn test_too_many_parties() {
        let session = MpcSession::new(
            ProofType::Range,
            ProofConfig::default(),
            MpcConfig {
                total_parties: 2,
                ..Default::default()
            },
        );

        for i in 1..=2 {
            let party = PartyInfo {
                id: PartyId::new(&format!("party-{}", i)),
                index: i,
                public_key: vec![0u8; 32],
                endpoint: None,
                status: PartyStatus::Pending,
            };
            session.add_party(party).await.unwrap();
        }

        let extra = PartyInfo {
            id: PartyId::new("party-3"),
            index: 3,
            public_key: vec![0u8; 32],
            endpoint: None,
            status: PartyStatus::Pending,
        };

        assert!(matches!(
            session.add_party(extra).await,
            Err(MpcError::TooManyParties)
        ));
    }

    #[test]
    fn test_shamir_sharing() {
        let config = MpcConfig {
            threshold: 2,
            total_parties: 3,
            sharing_scheme: SharingScheme::Shamir,
            ..Default::default()
        };

        let session = MpcSession::new(ProofType::Range, ProofConfig::default(), config);

        let secret = b"test secret data";
        let shares = session.generate_shares(secret);

        assert_eq!(shares.len(), 3);

        // Verify commitments are different
        let commitments: Vec<_> = shares.iter().map(|s| s.commitment).collect();
        assert_ne!(commitments[0], commitments[1]);
    }

    #[test]
    fn test_additive_sharing() {
        let config = MpcConfig {
            threshold: 3,
            total_parties: 3,
            sharing_scheme: SharingScheme::Additive,
            ..Default::default()
        };

        let session = MpcSession::new(ProofType::Range, ProofConfig::default(), config);

        let secret = b"test secret";
        let shares = session.generate_shares(secret);

        // Reconstruct
        let reconstructed = session.reconstruct(&shares).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_reconstruction_insufficient_shares() {
        let config = MpcConfig {
            threshold: 3,
            total_parties: 5,
            sharing_scheme: SharingScheme::Shamir,
            ..Default::default()
        };

        let session = MpcSession::new(ProofType::Range, ProofConfig::default(), config);

        let shares = vec![
            Share {
                holder: PartyId::new("party-1"),
                index: 1,
                value: vec![1, 2, 3],
                commitment: [0u8; 32],
            },
        ];

        assert!(matches!(
            session.reconstruct(&shares),
            Err(MpcError::InsufficientShares)
        ));
    }

    #[tokio::test]
    async fn test_session_state() {
        let session = MpcSession::new(
            ProofType::Range,
            ProofConfig::default(),
            MpcConfig::default(),
        );

        let state = session.get_state().await;
        assert_eq!(state.phase, MpcPhase::Setup);
        assert_eq!(state.current_round, 0);
    }

    #[tokio::test]
    async fn test_receive_share() {
        let session = MpcSession::new(
            ProofType::Range,
            ProofConfig::default(),
            MpcConfig::default(),
        );

        // Add party first
        let party = PartyInfo {
            id: PartyId::new("party-1"),
            index: 1,
            public_key: vec![0u8; 32],
            endpoint: None,
            status: PartyStatus::Ready,
        };
        session.add_party(party).await.unwrap();

        let share = Share {
            holder: PartyId::new("party-1"),
            index: 1,
            value: vec![1, 2, 3],
            commitment: [0u8; 32],
        };

        let msg = MpcMessage {
            from: PartyId::new("party-1"),
            msg_type: MpcMessageType::Share(share),
            payload: Vec::new(),
            timestamp: 0,
        };

        session.receive_message(msg).await.unwrap();

        let state = session.get_state().await;
        assert_eq!(state.shares_received, 1);
    }

    #[tokio::test]
    async fn test_unknown_party_message() {
        let session = MpcSession::new(
            ProofType::Range,
            ProofConfig::default(),
            MpcConfig::default(),
        );

        let msg = MpcMessage {
            from: PartyId::new("unknown"),
            msg_type: MpcMessageType::Ready,
            payload: Vec::new(),
            timestamp: 0,
        };

        assert!(matches!(
            session.receive_message(msg).await,
            Err(MpcError::UnknownParty(_))
        ));
    }

    #[tokio::test]
    async fn test_coordinator_stats() {
        let coordinator = MpcCoordinator::new(MpcConfig::default());

        let _ = coordinator
            .create_session(ProofType::Range, ProofConfig::default())
            .await;
        let _ = coordinator
            .create_session(ProofType::GradientIntegrity, ProofConfig::default())
            .await;

        let stats = coordinator.stats().await;
        assert_eq!(stats.sessions_created, 2);
    }

    #[test]
    fn test_sharing_schemes() {
        assert_ne!(SharingScheme::Shamir, SharingScheme::Additive);
        assert_ne!(SharingScheme::Additive, SharingScheme::Replicated);
    }

    #[test]
    fn test_party_status() {
        assert_ne!(PartyStatus::Pending, PartyStatus::Ready);
        assert_ne!(PartyStatus::Computing, PartyStatus::Done);
    }
}
