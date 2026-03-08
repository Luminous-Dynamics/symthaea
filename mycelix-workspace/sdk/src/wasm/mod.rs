//! # WebAssembly Bindings
//!
//! WASM bindings for the Mycelix SDK, enabling TypeScript/JavaScript to call
//! Rust code directly via WebAssembly.
//!
//! ## Building
//!
//! ```bash
//! # Install wasm-pack
//! cargo install wasm-pack
//!
//! # Build WASM package
//! wasm-pack build --target web --features wasm
//! ```
//!
//! ## Usage in TypeScript
//!
//! ```typescript
//! import init, { computeTrustScore, KVector } from '@mycelix/sdk-wasm';
//!
//! await init();
//!
//! const kvector = new KVector(0.8, 0.6, 0.9, 0.7, 0.5, 0.6, 0.7, 0.4, 0.8, 0.7);
//! const trust = computeTrustScore(kvector);
//! console.log(`Trust score: ${trust}`);
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

// ============================================================================
// K-Vector WASM Bindings
// ============================================================================

/// K-Vector values exposed to WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmKVector {
    k_r: f64,
    k_a: f64,
    k_i: f64,
    k_p: f64,
    k_m: f64,
    k_s: f64,
    k_h: f64,
    k_topo: f64,
    k_v: f64,
    k_coherence: f64,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmKVector {
    /// Create a new K-Vector
    #[wasm_bindgen(constructor)]
    pub fn new(
        k_r: f64,
        k_a: f64,
        k_i: f64,
        k_p: f64,
        k_m: f64,
        k_s: f64,
        k_h: f64,
        k_topo: f64,
        k_v: f64,
        k_coherence: f64,
    ) -> WasmKVector {
        WasmKVector {
            k_r: k_r.clamp(0.0, 1.0),
            k_a: k_a.clamp(0.0, 1.0),
            k_i: k_i.clamp(0.0, 1.0),
            k_p: k_p.clamp(0.0, 1.0),
            k_m: k_m.clamp(0.0, 1.0),
            k_s: k_s.clamp(0.0, 1.0),
            k_h: k_h.clamp(0.0, 1.0),
            k_topo: k_topo.clamp(0.0, 1.0),
            k_v: k_v.clamp(0.0, 1.0),
            k_coherence: k_coherence.clamp(0.0, 1.0),
        }
    }

    /// Create with default values
    #[wasm_bindgen(js_name = default)]
    pub fn default_kvector() -> WasmKVector {
        WasmKVector {
            k_r: 0.5,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
            k_v: 0.5,
            k_coherence: 0.5,
        }
    }

    /// Get reputation dimension
    #[wasm_bindgen(getter)]
    pub fn k_r(&self) -> f64 {
        self.k_r
    }

    /// Set reputation dimension
    #[wasm_bindgen(setter)]
    pub fn set_k_r(&mut self, value: f64) {
        self.k_r = value.clamp(0.0, 1.0);
    }

    /// Get activity dimension
    #[wasm_bindgen(getter)]
    pub fn k_a(&self) -> f64 {
        self.k_a
    }

    /// Set activity dimension
    #[wasm_bindgen(setter)]
    pub fn set_k_a(&mut self, value: f64) {
        self.k_a = value.clamp(0.0, 1.0);
    }

    /// Get integrity dimension
    #[wasm_bindgen(getter)]
    pub fn k_i(&self) -> f64 {
        self.k_i
    }

    /// Get performance dimension
    #[wasm_bindgen(getter)]
    pub fn k_p(&self) -> f64 {
        self.k_p
    }

    /// Get membership dimension
    #[wasm_bindgen(getter)]
    pub fn k_m(&self) -> f64 {
        self.k_m
    }

    /// Get stake dimension
    #[wasm_bindgen(getter)]
    pub fn k_s(&self) -> f64 {
        self.k_s
    }

    /// Get historical dimension
    #[wasm_bindgen(getter)]
    pub fn k_h(&self) -> f64 {
        self.k_h
    }

    /// Get topology dimension
    #[wasm_bindgen(getter)]
    pub fn k_topo(&self) -> f64 {
        self.k_topo
    }

    /// Get verification dimension
    #[wasm_bindgen(getter)]
    pub fn k_v(&self) -> f64 {
        self.k_v
    }

    /// Get phi/coherence dimension
    #[wasm_bindgen(getter)]
    pub fn k_coherence(&self) -> f64 {
        self.k_coherence
    }

    /// Convert to JSON string
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"k_r":{},"k_a":{},"k_i":{},"k_p":{},"k_m":{},"k_s":{},"k_h":{},"k_topo":{},"k_v":{},"k_coherence":{}}}"#,
            self.k_r,
            self.k_a,
            self.k_i,
            self.k_p,
            self.k_m,
            self.k_s,
            self.k_h,
            self.k_topo,
            self.k_v,
            self.k_coherence
        )
    }
}

// ============================================================================
// Trust Score Functions
// ============================================================================

/// Compute trust score from K-Vector
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = computeTrustScore)]
pub fn wasm_compute_trust_score(kvector: &WasmKVector) -> f64 {
    // Default weights
    let weights = [
        (kvector.k_r, 0.20),          // Reputation
        (kvector.k_a, 0.10),          // Activity
        (kvector.k_i, 0.20),          // Integrity
        (kvector.k_p, 0.15),          // Performance
        (kvector.k_m, 0.10),          // Membership
        (kvector.k_s, 0.05),          // Stake
        (kvector.k_h, 0.10),          // Historical
        (kvector.k_topo, 0.05),       // Topology
        (kvector.k_v, 0.025),         // Verification
        (kvector.k_coherence, 0.025), // Coherence
    ];

    let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
    let score: f64 = weights.iter().map(|(v, w)| v * w).sum();

    score / total_weight
}

/// Compute trust score with custom weights
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = computeTrustScoreWithWeights)]
pub fn wasm_compute_trust_score_weighted(
    kvector: &WasmKVector,
    w_r: f64,
    w_a: f64,
    w_i: f64,
    w_p: f64,
    w_m: f64,
    w_s: f64,
    w_h: f64,
    w_topo: f64,
    w_v: f64,
    w_phi: f64,
) -> f64 {
    let weights = [
        (kvector.k_r, w_r),
        (kvector.k_a, w_a),
        (kvector.k_i, w_i),
        (kvector.k_p, w_p),
        (kvector.k_m, w_m),
        (kvector.k_s, w_s),
        (kvector.k_h, w_h),
        (kvector.k_topo, w_topo),
        (kvector.k_v, w_v),
        (kvector.k_coherence, w_phi),
    ];

    let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
    if total_weight == 0.0 {
        return 0.0;
    }

    let score: f64 = weights.iter().map(|(v, w)| v * w).sum();
    score / total_weight
}

/// Calculate KREDIT cap from trust score
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = calculateKreditFromTrust)]
pub fn wasm_calculate_kredit_from_trust(
    trust_score: f64,
    base_kredit: u64,
    max_multiplier: f64,
) -> u64 {
    let clamped_trust = trust_score.clamp(0.0, 1.0);
    let multiplier = 1.0 + (max_multiplier - 1.0) * clamped_trust;
    (base_kredit as f64 * multiplier) as u64
}

// ============================================================================
// Coordination WASM Bindings
// ============================================================================

/// Vote type for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum WasmVoteType {
    Approve,
    Reject,
    Abstain,
}

/// Consensus decision for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum WasmConsensusDecision {
    Approved,
    Rejected,
    NoQuorum,
    Pending,
    Expired,
}

/// Agent group for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmAgentGroup {
    members: Vec<(String, f64)>,
    proposals: Vec<WasmProposal>,
    votes: Vec<WasmVote>,
    config: WasmCoordinationConfig,
}

#[cfg(feature = "wasm")]
#[derive(Clone)]
struct WasmProposal {
    id: String,
    action: String,
    description: String,
    created_at: u64,
    expires_at: u64,
    finalized: bool,
}

#[cfg(feature = "wasm")]
#[derive(Clone)]
struct WasmVote {
    proposal_id: String,
    agent_id: String,
    vote_type: WasmVoteType,
    weight: f64,
}

#[cfg(feature = "wasm")]
#[derive(Clone)]
struct WasmCoordinationConfig {
    min_trust_threshold: f64,
    approval_threshold: f64,
    min_participation: f64,
    quadratic_voting: bool,
}

#[cfg(feature = "wasm")]
impl Default for WasmCoordinationConfig {
    fn default() -> Self {
        Self {
            min_trust_threshold: 0.3,
            approval_threshold: 0.67,
            min_participation: 0.5,
            quadratic_voting: false,
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmAgentGroup {
    /// Create a new agent group
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAgentGroup {
        WasmAgentGroup {
            members: Vec::new(),
            proposals: Vec::new(),
            votes: Vec::new(),
            config: WasmCoordinationConfig::default(),
        }
    }

    /// Create with custom config
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(
        min_trust: f64,
        approval_threshold: f64,
        min_participation: f64,
        quadratic_voting: bool,
    ) -> WasmAgentGroup {
        WasmAgentGroup {
            members: Vec::new(),
            proposals: Vec::new(),
            votes: Vec::new(),
            config: WasmCoordinationConfig {
                min_trust_threshold: min_trust,
                approval_threshold,
                min_participation,
                quadratic_voting,
            },
        }
    }

    /// Add a member
    #[wasm_bindgen(js_name = addMember)]
    pub fn add_member(&mut self, agent_id: &str, trust: f64) -> bool {
        if trust < self.config.min_trust_threshold {
            return false;
        }
        if self.members.iter().any(|(id, _)| id == agent_id) {
            return false;
        }
        self.members
            .push((agent_id.to_string(), trust.clamp(0.0, 1.0)));
        true
    }

    /// Remove a member
    #[wasm_bindgen(js_name = removeMember)]
    pub fn remove_member(&mut self, agent_id: &str) -> bool {
        let len_before = self.members.len();
        self.members.retain(|(id, _)| id != agent_id);
        self.members.len() < len_before
    }

    /// Get member count
    #[wasm_bindgen(js_name = memberCount)]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get total trust
    #[wasm_bindgen(js_name = totalTrust)]
    pub fn total_trust(&self) -> f64 {
        self.members.iter().map(|(_, t)| t).sum()
    }

    /// Submit a proposal
    #[wasm_bindgen(js_name = submitProposal)]
    pub fn submit_proposal(&mut self, action: &str, description: &str) -> String {
        let now = js_sys_now();
        let id = format!("prop-{}", now);

        self.proposals.push(WasmProposal {
            id: id.clone(),
            action: action.to_string(),
            description: description.to_string(),
            created_at: now,
            expires_at: now + 3600_000,
            finalized: false,
        });

        id
    }

    /// Cast a vote
    #[wasm_bindgen]
    pub fn vote(&mut self, agent_id: &str, proposal_id: &str, vote_type: WasmVoteType) -> bool {
        // Check membership
        let trust = match self.members.iter().find(|(id, _)| id == agent_id) {
            Some((_, t)) => *t,
            None => return false,
        };

        // Check proposal exists and not finalized
        let proposal = match self.proposals.iter().find(|p| p.id == proposal_id) {
            Some(p) if !p.finalized => p,
            _ => return false,
        };

        // Calculate weight
        let weight = if self.config.quadratic_voting {
            trust.sqrt()
        } else {
            trust
        };

        // Remove existing vote if any
        self.votes
            .retain(|v| !(v.proposal_id == proposal_id && v.agent_id == agent_id));

        // Add vote
        self.votes.push(WasmVote {
            proposal_id: proposal_id.to_string(),
            agent_id: agent_id.to_string(),
            vote_type,
            weight,
        });

        true
    }

    /// Check consensus (returns JSON result)
    #[wasm_bindgen(js_name = checkConsensus)]
    pub fn check_consensus(&mut self, proposal_id: &str) -> String {
        // First check if proposal exists (immutable borrow)
        if !self.proposals.iter().any(|p| p.id == proposal_id) {
            return r#"{"error":"Proposal not found"}"#.to_string();
        }

        // Calculate vote weights (immutable borrow of votes)
        let mut approval_weight = 0.0;
        let mut rejection_weight = 0.0;
        let mut abstention_weight = 0.0;

        for vote in self.votes.iter().filter(|v| v.proposal_id == proposal_id) {
            match vote.vote_type {
                WasmVoteType::Approve => approval_weight += vote.weight,
                WasmVoteType::Reject => rejection_weight += vote.weight,
                WasmVoteType::Abstain => abstention_weight += vote.weight,
            }
        }

        let total_vote_weight = approval_weight + rejection_weight + abstention_weight;

        // Calculate total possible (needs immutable self access)
        let total_possible = if self.config.quadratic_voting {
            self.members.iter().map(|(_, t)| t.sqrt()).sum()
        } else {
            self.total_trust()
        };

        let participation_rate = if total_possible > 0.0 {
            total_vote_weight / total_possible
        } else {
            0.0
        };

        let quorum_reached = participation_rate >= self.config.min_participation;

        // Determine decision
        let decision = if !quorum_reached {
            "pending"
        } else {
            let active_weight = approval_weight + rejection_weight;
            if active_weight > 0.0 {
                let approval_ratio = approval_weight / active_weight;
                if approval_ratio >= self.config.approval_threshold {
                    "approved"
                } else {
                    "rejected"
                }
            } else {
                "pending"
            }
        };

        // Now get mutable reference to finalize if needed
        if decision == "approved" || decision == "rejected" {
            if let Some(proposal) = self.proposals.iter_mut().find(|p| p.id == proposal_id) {
                proposal.finalized = true;
            }
        }

        format!(
            r#"{{"proposalId":"{}","decision":"{}","approvalWeight":{},"rejectionWeight":{},"abstentionWeight":{},"participationRate":{},"quorumReached":{}}}"#,
            proposal_id,
            decision,
            approval_weight,
            rejection_weight,
            abstention_weight,
            participation_rate,
            quorum_reached
        )
    }
}

#[cfg(feature = "wasm")]
fn js_sys_now() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as u64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get SDK version
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = sdkVersion)]
pub fn wasm_sdk_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Validate DID format
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = isValidDid)]
pub fn wasm_is_valid_did(did: &str) -> bool {
    did.starts_with("did:") && did.len() > 10 && did.len() < 256
}

// ============================================================================
// Tests (non-WASM)
// ============================================================================

#[cfg(test)]
mod tests {
    // Tests for the non-WASM versions of these functions would go here
    // WASM tests are typically run via wasm-pack test

    #[test]
    fn test_module_exists() {
        // Basic sanity test that the module compiles
        assert!(true);
    }
}
