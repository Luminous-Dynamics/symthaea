use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Consciousness attestation types
// ---------------------------------------------------------------------------

/// Data passed from host to guest (zkVM).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionInput {
    /// HDC episode vectors (e.g., 1024D each).
    pub episodes: Vec<Vec<f32>>,
    /// Temporal scaling factor (CfC tau).
    pub tau_scale: f32,
    /// Phi threshold for attestation pass/fail.
    pub threshold: f32,
}

/// Data committed by the guest as public output.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvolutionOutput {
    /// Mean Phi across all episodes.
    pub average_phi: f32,
    /// The tau_scale that was used.
    pub tau_scale: f32,
    /// Number of episodes processed.
    pub episode_count: u32,
}

// ---------------------------------------------------------------------------
// Financial balance proof types
// ---------------------------------------------------------------------------

/// Input for a balance sufficiency proof.
/// Proves: balance >= required_amount WITHOUT revealing the actual balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceProofInput {
    /// The actual balance (PRIVATE -- not revealed in proof).
    pub balance: u64,
    /// The required minimum (PUBLIC -- included in proof).
    pub required_minimum: u64,
    /// Nonce to prevent replay (PUBLIC).
    pub nonce: u64,
}

/// Output of a balance sufficiency proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceProofOutput {
    /// Whether balance >= required_minimum (PUBLIC).
    pub sufficient: bool,
    /// The required minimum that was checked (PUBLIC).
    pub required_minimum: u64,
    /// Nonce (PUBLIC).
    pub nonce: u64,
}
