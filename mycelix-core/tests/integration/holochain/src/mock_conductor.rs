// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mock Conductor for CI Testing
//!
//! Provides a SweetTest-compatible API that works without a real Holochain conductor.
//! This allows running tests in CI while validating zome call patterns.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use mycelix_holochain_tests::{MockConductor, MockZome};
//!
//! #[tokio::test]
//! async fn test_fl_round() {
//!     let conductor = MockConductor::new();
//!     let fl_zome = conductor.get_zome("federated_learning");
//!
//!     // Submit gradient
//!     let result = fl_zome.call("submit_gradient", SubmitGradientInput {
//!         round: 1,
//!         gradient: vec![1.0, 2.0, 3.0],
//!     }).await;
//!
//!     assert!(result.is_ok());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Result of a zome call
pub type ZomeCallResult<T> = Result<T, ZomeCallError>;

/// Error from a zome call
#[derive(Debug, Clone)]
pub enum ZomeCallError {
    /// Zome not found
    ZomeNotFound(String),
    /// Function not found
    FunctionNotFound(String),
    /// Validation failed
    ValidationFailed(String),
    /// Deserialization error
    DeserializationError(String),
    /// Conductor error
    ConductorError(String),
    /// Network error (for multi-agent tests)
    NetworkError(String),
}

impl std::fmt::Display for ZomeCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZomeCallError::ZomeNotFound(z) => write!(f, "Zome not found: {}", z),
            ZomeCallError::FunctionNotFound(fn_name) => write!(f, "Function not found: {}", fn_name),
            ZomeCallError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            ZomeCallError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            ZomeCallError::ConductorError(msg) => write!(f, "Conductor error: {}", msg),
            ZomeCallError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl std::error::Error for ZomeCallError {}

/// Mock agent public key (simulates AgentPubKey)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MockAgentPubKey(pub String);

impl MockAgentPubKey {
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 16] = rng.gen();
        Self(format!("uhCAk_{}", hex::encode(&bytes)))
    }

    pub fn from_str(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Hex encoding helper
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// Mock entry hash
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MockEntryHash(pub String);

impl MockEntryHash {
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 32] = rng.gen();
        Self(format!("uhCEk_{}", hex::encode(&bytes)))
    }
}

/// Mock action hash
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MockActionHash(pub String);

impl MockActionHash {
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 32] = rng.gen();
        Self(format!("uhCkk_{}", hex::encode(&bytes)))
    }
}

/// Mock conductor that simulates Holochain conductor behavior
pub struct MockConductor {
    /// Registered zomes
    zomes: HashMap<String, MockZome>,
    /// Agents in the conductor
    agents: Vec<MockAgentPubKey>,
    /// Cells (agent + DNA combinations)
    cells: Vec<MockCell>,
    /// DHT state (shared across cells)
    dht: Arc<Mutex<MockDht>>,
}

impl MockConductor {
    /// Create a new mock conductor
    pub fn new() -> Self {
        Self {
            zomes: HashMap::new(),
            agents: Vec::new(),
            cells: Vec::new(),
            dht: Arc::new(Mutex::new(MockDht::new())),
        }
    }

    /// Create conductor with standard Mycelix zomes
    pub fn with_mycelix_zomes() -> Self {
        let mut conductor = Self::new();

        // Register standard zomes with their mock implementations
        conductor.register_zome(MockZome::federated_learning());
        conductor.register_zome(MockZome::bridge());
        conductor.register_zome(MockZome::pogq_validation());
        conductor.register_zome(MockZome::agents());

        conductor
    }

    /// Register a zome
    pub fn register_zome(&mut self, zome: MockZome) {
        self.zomes.insert(zome.name.clone(), zome);
    }

    /// Add an agent to the conductor
    pub fn add_agent(&mut self) -> MockAgentPubKey {
        let agent = MockAgentPubKey::random();
        self.agents.push(agent.clone());

        // Create a cell for this agent with all registered zomes
        let cell = MockCell {
            agent: agent.clone(),
            zomes: self.zomes.keys().cloned().collect(),
            dht: Arc::clone(&self.dht),
        };
        self.cells.push(cell);

        agent
    }

    /// Get a cell by agent
    pub fn get_cell(&self, agent: &MockAgentPubKey) -> Option<&MockCell> {
        self.cells.iter().find(|c| &c.agent == agent)
    }

    /// Get a zome by name
    pub fn get_zome(&self, name: &str) -> Option<&MockZome> {
        self.zomes.get(name)
    }

    /// Call a zome function (convenience method)
    pub async fn call<I: Serialize, O: for<'de> Deserialize<'de>>(
        &self,
        agent: &MockAgentPubKey,
        zome: &str,
        function: &str,
        input: I,
    ) -> ZomeCallResult<O> {
        let cell = self.get_cell(agent)
            .ok_or_else(|| ZomeCallError::ConductorError("Agent not found".to_string()))?;

        cell.call(zome, function, input).await
    }

    /// Wait for DHT consistency (no-op in mock)
    pub async fn consistency(&self, _timeout_secs: u64) {
        // Mock: instant consistency
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

impl Default for MockConductor {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock cell (agent + DNA)
pub struct MockCell {
    /// Agent public key
    pub agent: MockAgentPubKey,
    /// Available zomes
    pub zomes: Vec<String>,
    /// Shared DHT reference
    dht: Arc<Mutex<MockDht>>,
}

impl MockCell {
    /// Call a zome function
    pub async fn call<I: Serialize, O: for<'de> Deserialize<'de>>(
        &self,
        zome: &str,
        function: &str,
        input: I,
    ) -> ZomeCallResult<O> {
        if !self.zomes.contains(&zome.to_string()) {
            return Err(ZomeCallError::ZomeNotFound(zome.to_string()));
        }

        // Serialize input
        let input_json = serde_json::to_value(input)
            .map_err(|e| ZomeCallError::DeserializationError(e.to_string()))?;

        // Execute mock function
        let result_json = self.execute_function(zome, function, input_json).await?;

        // Deserialize output
        serde_json::from_value(result_json)
            .map_err(|e| ZomeCallError::DeserializationError(e.to_string()))
    }

    /// Execute a mock zome function
    async fn execute_function(
        &self,
        zome: &str,
        function: &str,
        input: serde_json::Value,
    ) -> ZomeCallResult<serde_json::Value> {
        match (zome, function) {
            // Federated Learning zome
            ("federated_learning", "submit_gradient") => {
                self.fl_submit_gradient(input).await
            }
            ("federated_learning", "get_round_gradients") => {
                self.fl_get_round_gradients(input).await
            }
            ("federated_learning", "aggregate_round") => {
                self.fl_aggregate_round(input).await
            }

            // Bridge zome
            ("bridge", "register_happ") => {
                self.bridge_register_happ(input).await
            }
            ("bridge", "share_reputation") => {
                self.bridge_share_reputation(input).await
            }
            ("bridge", "get_cross_happ_reputation") => {
                self.bridge_get_reputation(input).await
            }

            // PoGQ validation zome
            ("pogq_validation", "publish_proof") => {
                self.pogq_publish_proof(input).await
            }
            ("pogq_validation", "verify_proof") => {
                self.pogq_verify_proof(input).await
            }

            // Agents zome
            ("agents", "register_agent") => {
                self.agents_register(input).await
            }
            ("agents", "get_agent_info") => {
                self.agents_get_info(input).await
            }

            _ => Err(ZomeCallError::FunctionNotFound(format!("{}::{}", zome, function)))
        }
    }

    // ==================== FL Zome Mock Implementations ====================

    async fn fl_submit_gradient(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let round = input.get("round").and_then(|v| v.as_u64()).unwrap_or(1);
        let gradient = input.get("gradient").cloned().unwrap_or(serde_json::json!([]));

        // Store in DHT
        let action_hash = MockActionHash::random();
        {
            let mut dht = self.dht.lock().unwrap();
            dht.entries.insert(
                action_hash.0.clone(),
                serde_json::json!({
                    "type": "gradient",
                    "round": round,
                    "gradient": gradient,
                    "agent": self.agent.0,
                }),
            );
        }

        Ok(serde_json::json!({
            "action_hash": action_hash.0,
            "round": round,
        }))
    }

    async fn fl_get_round_gradients(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let round = input.get("round").and_then(|v| v.as_u64()).unwrap_or(1);

        let dht = self.dht.lock().unwrap();
        let gradients: Vec<_> = dht.entries.values()
            .filter(|e| {
                e.get("type") == Some(&serde_json::json!("gradient"))
                    && e.get("round") == Some(&serde_json::json!(round))
            })
            .cloned()
            .collect();

        Ok(serde_json::json!({ "gradients": gradients }))
    }

    async fn fl_aggregate_round(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let round = input.get("round").and_then(|v| v.as_u64()).unwrap_or(1);

        // Simple mock aggregation
        Ok(serde_json::json!({
            "round": round,
            "aggregated": [0.5, 0.5, 0.5],
            "contributors": 3,
            "status": "completed",
        }))
    }

    // ==================== Bridge Zome Mock Implementations ====================

    async fn bridge_register_happ(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let happ_id = input.get("happ_id").and_then(|v| v.as_str()).unwrap_or("unknown");

        {
            let mut dht = self.dht.lock().unwrap();
            dht.entries.insert(
                format!("happ:{}", happ_id),
                serde_json::json!({
                    "type": "happ_registration",
                    "happ_id": happ_id,
                    "registrant": self.agent.0,
                }),
            );
        }

        Ok(serde_json::json!({
            "happ_id": happ_id,
            "registered": true,
        }))
    }

    async fn bridge_share_reputation(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let target_agent = input.get("agent").and_then(|v| v.as_str()).unwrap_or("");
        let score = input.get("score").and_then(|v| v.as_f64()).unwrap_or(0.5);

        {
            let mut dht = self.dht.lock().unwrap();
            dht.entries.insert(
                format!("rep:{}:{}", target_agent, self.agent.0),
                serde_json::json!({
                    "type": "reputation",
                    "agent": target_agent,
                    "score": score,
                    "source": self.agent.0,
                }),
            );
        }

        Ok(serde_json::json!({
            "shared": true,
            "agent": target_agent,
            "score": score,
        }))
    }

    async fn bridge_get_reputation(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let target_agent = input.get("agent").and_then(|v| v.as_str()).unwrap_or("");

        let dht = self.dht.lock().unwrap();
        let reps: Vec<f64> = dht.entries.values()
            .filter(|e| {
                e.get("type") == Some(&serde_json::json!("reputation"))
                    && e.get("agent") == Some(&serde_json::json!(target_agent))
            })
            .filter_map(|e| e.get("score").and_then(|s| s.as_f64()))
            .collect();

        let aggregate = if reps.is_empty() {
            0.5
        } else {
            reps.iter().sum::<f64>() / reps.len() as f64
        };

        Ok(serde_json::json!({
            "agent": target_agent,
            "aggregate_score": aggregate,
            "num_sources": reps.len(),
        }))
    }

    // ==================== PoGQ Zome Mock Implementations ====================

    async fn pogq_publish_proof(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let round = input.get("round").and_then(|v| v.as_u64()).unwrap_or(1);
        let proof_bytes = input.get("proof").cloned().unwrap_or(serde_json::json!([]));

        let action_hash = MockActionHash::random();
        {
            let mut dht = self.dht.lock().unwrap();
            dht.entries.insert(
                action_hash.0.clone(),
                serde_json::json!({
                    "type": "pogq_proof",
                    "round": round,
                    "proof": proof_bytes,
                    "agent": self.agent.0,
                    "verified": false,
                }),
            );
        }

        Ok(serde_json::json!({
            "action_hash": action_hash.0,
            "published": true,
        }))
    }

    async fn pogq_verify_proof(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let proof_hash = input.get("proof_hash").and_then(|v| v.as_str()).unwrap_or("");

        let mut dht = self.dht.lock().unwrap();
        if let Some(entry) = dht.entries.get_mut(proof_hash) {
            if entry.get("type") == Some(&serde_json::json!("pogq_proof")) {
                entry["verified"] = serde_json::json!(true);
                return Ok(serde_json::json!({
                    "valid": true,
                    "proof_hash": proof_hash,
                }));
            }
        }

        Ok(serde_json::json!({
            "valid": false,
            "error": "Proof not found",
        }))
    }

    // ==================== Agents Zome Mock Implementations ====================

    async fn agents_register(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let name = input.get("name").and_then(|v| v.as_str()).unwrap_or("Anonymous");

        {
            let mut dht = self.dht.lock().unwrap();
            dht.entries.insert(
                format!("agent:{}", self.agent.0),
                serde_json::json!({
                    "type": "agent_profile",
                    "agent": self.agent.0,
                    "name": name,
                    "registered_at": chrono::Utc::now().timestamp(),
                }),
            );
        }

        Ok(serde_json::json!({
            "registered": true,
            "agent": self.agent.0,
        }))
    }

    async fn agents_get_info(&self, input: serde_json::Value) -> ZomeCallResult<serde_json::Value> {
        let agent_key = input.get("agent").and_then(|v| v.as_str()).unwrap_or(&self.agent.0);

        let dht = self.dht.lock().unwrap();
        if let Some(entry) = dht.entries.get(&format!("agent:{}", agent_key)) {
            return Ok(entry.clone());
        }

        Ok(serde_json::json!({
            "agent": agent_key,
            "name": "Unknown",
            "registered": false,
        }))
    }
}

/// Mock DHT storage
struct MockDht {
    /// Entries by hash
    entries: HashMap<String, serde_json::Value>,
    /// Links
    links: HashMap<String, Vec<String>>,
}

impl MockDht {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            links: HashMap::new(),
        }
    }
}

/// Mock zome definition
pub struct MockZome {
    /// Zome name
    pub name: String,
    /// Available functions
    pub functions: Vec<String>,
}

impl MockZome {
    /// Create federated_learning zome
    pub fn federated_learning() -> Self {
        Self {
            name: "federated_learning".to_string(),
            functions: vec![
                "submit_gradient".to_string(),
                "get_round_gradients".to_string(),
                "aggregate_round".to_string(),
                "get_my_gradients".to_string(),
                "get_round_status".to_string(),
            ],
        }
    }

    /// Create bridge zome
    pub fn bridge() -> Self {
        Self {
            name: "bridge".to_string(),
            functions: vec![
                "register_happ".to_string(),
                "share_reputation".to_string(),
                "get_cross_happ_reputation".to_string(),
                "broadcast_event".to_string(),
                "verify_credential".to_string(),
            ],
        }
    }

    /// Create pogq_validation zome
    pub fn pogq_validation() -> Self {
        Self {
            name: "pogq_validation".to_string(),
            functions: vec![
                "publish_proof".to_string(),
                "verify_proof".to_string(),
                "get_round_proofs".to_string(),
                "check_quarantine_status".to_string(),
            ],
        }
    }

    /// Create agents zome
    pub fn agents() -> Self {
        Self {
            name: "agents".to_string(),
            functions: vec![
                "register_agent".to_string(),
                "get_agent_info".to_string(),
                "update_profile".to_string(),
                "list_agents".to_string(),
            ],
        }
    }
}

// Timestamp helper
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime { DateTime }
    }
    pub struct DateTime;
    impl DateTime {
        pub fn timestamp(&self) -> i64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_conductor_creation() {
        let conductor = MockConductor::with_mycelix_zomes();
        assert!(conductor.get_zome("federated_learning").is_some());
        assert!(conductor.get_zome("bridge").is_some());
    }

    #[tokio::test]
    async fn test_add_agent() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();
        assert!(conductor.get_cell(&agent).is_some());
    }

    #[tokio::test]
    async fn test_fl_submit_gradient() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        let result: serde_json::Value = conductor.call(
            &agent,
            "federated_learning",
            "submit_gradient",
            serde_json::json!({
                "round": 1,
                "gradient": [0.1, 0.2, 0.3],
            }),
        ).await.unwrap();

        assert!(result.get("action_hash").is_some());
        assert_eq!(result.get("round"), Some(&serde_json::json!(1)));
    }

    #[tokio::test]
    async fn test_bridge_reputation_sharing() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let alice = conductor.add_agent();
        let bob = conductor.add_agent();

        // Alice shares reputation for Bob
        let _: serde_json::Value = conductor.call(
            &alice,
            "bridge",
            "share_reputation",
            serde_json::json!({
                "agent": bob.0,
                "score": 0.9,
            }),
        ).await.unwrap();

        // Get Bob's reputation
        let rep: serde_json::Value = conductor.call(
            &bob,
            "bridge",
            "get_cross_happ_reputation",
            serde_json::json!({ "agent": bob.0 }),
        ).await.unwrap();

        assert_eq!(rep.get("aggregate_score"), Some(&serde_json::json!(0.9)));
    }

    #[tokio::test]
    async fn test_pogq_proof_lifecycle() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Publish proof
        let publish_result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0xde, 0xad, 0xbe, 0xef],
            }),
        ).await.unwrap();

        let proof_hash = publish_result.get("action_hash").unwrap().as_str().unwrap();

        // Verify proof
        let verify_result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "verify_proof",
            serde_json::json!({ "proof_hash": proof_hash }),
        ).await.unwrap();

        assert_eq!(verify_result.get("valid"), Some(&serde_json::json!(true)));
    }
}
