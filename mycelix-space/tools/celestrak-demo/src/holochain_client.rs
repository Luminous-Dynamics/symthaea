//! Holochain Client Module for Mycelix Space
//!
//! Provides connectivity to Holochain conductor for ingesting orbital data.
//! This module handles:
//! - WebSocket connection to conductor
//! - Zome function calls
//! - Entry creation for orbital objects, TLEs, and observations

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Holochain connection
#[derive(Debug, Clone)]
pub struct HolochainConfig {
    /// Conductor WebSocket URL (default: ws://localhost:8888)
    pub conductor_url: String,
    /// DNA hash to call
    pub dna_hash: Option<String>,
    /// Cell role name
    pub role_name: String,
    /// Agent public key (hex encoded)
    pub agent_key: Option<String>,
}

impl Default for HolochainConfig {
    fn default() -> Self {
        Self {
            conductor_url: "ws://localhost:8888".to_string(),
            dna_hash: None,
            role_name: "space_operator".to_string(),
            agent_key: None,
        }
    }
}

/// Orbital object entry for Holochain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalObjectInput {
    pub norad_id: u32,
    pub name: String,
    pub object_type: String,  // "Payload", "RocketBody", "Debris", "Unknown"
    pub launch_date: Option<DateTime<Utc>>,
    pub decay_date: Option<DateTime<Utc>>,
    pub owner_country: Option<String>,
    pub data_source: String,  // "SpaceTrack", "CelesTrak", "Operator", "Computed"
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// TLE entry for Holochain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TleInput {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub epoch: DateTime<Utc>,
    pub source: String,
}

/// State vector entry for Holochain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVectorInput {
    pub norad_id: u32,
    pub epoch: DateTime<Utc>,
    pub position_km: [f64; 3],
    pub velocity_kms: [f64; 3],
    pub covariance: Option<[f64; 21]>,
    pub reference_frame: String,  // "Teme", "J2000", "Itrf", "Gcrf"
    pub quality: f64,
    pub source: String,
}

/// Result from Holochain zome call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZomeCallResult {
    pub action_hash: String,
    pub entry_hash: String,
}

/// Holochain client for Mycelix Space
pub struct HolochainClient {
    config: HolochainConfig,
    connected: bool,
}

impl HolochainClient {
    /// Create a new Holochain client
    pub fn new(config: HolochainConfig) -> Self {
        Self {
            config,
            connected: false,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(HolochainConfig::default())
    }

    /// Connect to the conductor
    pub async fn connect(&mut self) -> Result<()> {
        // In a real implementation, this would establish a WebSocket connection
        // For now, we'll use the holochain_client crate patterns
        println!("Connecting to Holochain conductor at {}...", self.config.conductor_url);

        // TODO: Implement real WebSocket connection
        // let client = WebsocketClient::connect(&self.config.conductor_url).await?;

        self.connected = true;
        println!("Connection established (simulated)");
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Create an orbital object entry
    pub async fn create_orbital_object(&self, input: OrbitalObjectInput) -> Result<ZomeCallResult> {
        if !self.connected {
            anyhow::bail!("Not connected to Holochain conductor");
        }

        println!("Creating orbital object: NORAD {} - {}", input.norad_id, input.name);

        // Simulate zome call
        // In real implementation:
        // let result = self.call_zome("orbital_objects_coordinator", "create_orbital_object", input).await?;

        Ok(ZomeCallResult {
            action_hash: format!("uhCkk-SIMULATED-{}", input.norad_id),
            entry_hash: format!("uhCEk-SIMULATED-{}", input.norad_id),
        })
    }

    /// Create a TLE entry
    pub async fn create_tle(&self, input: TleInput) -> Result<ZomeCallResult> {
        if !self.connected {
            anyhow::bail!("Not connected to Holochain conductor");
        }

        println!("Creating TLE for NORAD {}", input.norad_id);

        Ok(ZomeCallResult {
            action_hash: format!("uhCkk-TLE-SIMULATED-{}", input.norad_id),
            entry_hash: format!("uhCEk-TLE-SIMULATED-{}", input.norad_id),
        })
    }

    /// Create a state vector entry
    pub async fn create_state_vector(&self, input: StateVectorInput) -> Result<ZomeCallResult> {
        if !self.connected {
            anyhow::bail!("Not connected to Holochain conductor");
        }

        println!("Creating state vector for NORAD {} at {}", input.norad_id, input.epoch);

        Ok(ZomeCallResult {
            action_hash: format!("uhCkk-SV-SIMULATED-{}", input.norad_id),
            entry_hash: format!("uhCEk-SV-SIMULATED-{}", input.norad_id),
        })
    }

    /// Batch create orbital objects
    pub async fn batch_create_objects(&self, objects: Vec<OrbitalObjectInput>) -> Result<Vec<ZomeCallResult>> {
        let mut results = Vec::new();
        for obj in objects {
            results.push(self.create_orbital_object(obj).await?);
        }
        Ok(results)
    }

    /// Get conductor URL
    pub fn conductor_url(&self) -> &str {
        &self.config.conductor_url
    }
}

/// Builder for batch ingestion
pub struct IngestionBatch {
    objects: Vec<OrbitalObjectInput>,
    tles: Vec<TleInput>,
    state_vectors: Vec<StateVectorInput>,
}

impl IngestionBatch {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            tles: Vec::new(),
            state_vectors: Vec::new(),
        }
    }

    pub fn add_object(mut self, object: OrbitalObjectInput) -> Self {
        self.objects.push(object);
        self
    }

    pub fn add_tle(mut self, tle: TleInput) -> Self {
        self.tles.push(tle);
        self
    }

    pub fn add_state_vector(mut self, sv: StateVectorInput) -> Self {
        self.state_vectors.push(sv);
        self
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    pub fn tle_count(&self) -> usize {
        self.tles.len()
    }

    pub fn state_vector_count(&self) -> usize {
        self.state_vectors.len()
    }

    pub fn objects(&self) -> &[OrbitalObjectInput] {
        &self.objects
    }

    pub fn tles(&self) -> &[TleInput] {
        &self.tles
    }

    pub fn state_vectors(&self) -> &[StateVectorInput] {
        &self.state_vectors
    }

    /// Ingest all data into Holochain
    pub async fn ingest(&self, client: &HolochainClient) -> Result<IngestionReport> {
        let mut report = IngestionReport::new();

        // Ingest objects
        for obj in &self.objects {
            match client.create_orbital_object(obj.clone()).await {
                Ok(result) => {
                    report.objects_created += 1;
                    report.action_hashes.push(result.action_hash);
                }
                Err(e) => {
                    report.errors.push(format!("Failed to create object {}: {}", obj.norad_id, e));
                }
            }
        }

        // Ingest TLEs
        for tle in &self.tles {
            match client.create_tle(tle.clone()).await {
                Ok(result) => {
                    report.tles_created += 1;
                    report.action_hashes.push(result.action_hash);
                }
                Err(e) => {
                    report.errors.push(format!("Failed to create TLE {}: {}", tle.norad_id, e));
                }
            }
        }

        // Ingest state vectors
        for sv in &self.state_vectors {
            match client.create_state_vector(sv.clone()).await {
                Ok(result) => {
                    report.state_vectors_created += 1;
                    report.action_hashes.push(result.action_hash);
                }
                Err(e) => {
                    report.errors.push(format!("Failed to create state vector {}: {}", sv.norad_id, e));
                }
            }
        }

        Ok(report)
    }
}

impl Default for IngestionBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Report from batch ingestion
#[derive(Debug, Default)]
pub struct IngestionReport {
    pub objects_created: usize,
    pub tles_created: usize,
    pub state_vectors_created: usize,
    pub action_hashes: Vec<String>,
    pub errors: Vec<String>,
}

impl IngestionReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_created(&self) -> usize {
        self.objects_created + self.tles_created + self.state_vectors_created
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_summary(&self) {
        println!("\n=== Ingestion Report ===");
        println!("Orbital Objects Created: {}", self.objects_created);
        println!("TLEs Created: {}", self.tles_created);
        println!("State Vectors Created: {}", self.state_vectors_created);
        println!("Total Entries: {}", self.total_created());

        if self.has_errors() {
            println!("\nErrors ({}):", self.errors.len());
            for err in &self.errors {
                println!("  - {}", err);
            }
        } else {
            println!("\nNo errors during ingestion.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingestion_batch() {
        let batch = IngestionBatch::new()
            .add_object(OrbitalObjectInput {
                norad_id: 25544,
                name: "ISS".to_string(),
                object_type: "Payload".to_string(),
                launch_date: None,
                decay_date: None,
                owner_country: Some("ISS".to_string()),
                data_source: "CelesTrak".to_string(),
                metadata: HashMap::new(),
            })
            .add_tle(TleInput {
                norad_id: 25544,
                line1: "1 25544U".to_string(),
                line2: "2 25544".to_string(),
                epoch: Utc::now(),
                source: "CelesTrak".to_string(),
            });

        assert_eq!(batch.object_count(), 1);
        assert_eq!(batch.tle_count(), 1);
        assert_eq!(batch.state_vector_count(), 0);
    }
}
