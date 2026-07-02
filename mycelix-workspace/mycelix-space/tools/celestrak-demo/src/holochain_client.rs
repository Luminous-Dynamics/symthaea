// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holochain Client Module for Mycelix Space
//!
//! Provides file-based ingestion into Holochain via JSON export.
//! Each "zome call" writes a JSON file matching the exact zome input types,
//! which can be imported via `hc sandbox call` or the batch import script.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Configuration for Holochain ingestion
#[derive(Debug, Clone)]
pub struct HolochainConfig {
    /// Output directory for JSON batch files
    pub batch_dir: PathBuf,
    /// Cell role name
    pub _role_name: String,
}

impl Default for HolochainConfig {
    fn default() -> Self {
        Self {
            batch_dir: PathBuf::from("import_batch"),
            _role_name: "space_operator".to_string(),
        }
    }
}

/// Input matching the `register_object` zome function exactly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterObjectCall {
    pub norad_id: u32,
    pub intl_designator: String,
    pub name: String,
    pub object_type: String,
    pub country: Option<String>,
    pub launch_date: Option<i64>,
    pub status: Option<String>,
}

/// Input matching the `submit_tle` zome function exactly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitTleCall {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub source: Option<String>,
    pub quality: Option<u8>,
}

/// Orbital object entry for building ingestion batches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrbitalObjectInput {
    pub norad_id: u32,
    pub name: String,
    pub object_type: String,
    pub launch_date: Option<DateTime<Utc>>,
    pub decay_date: Option<DateTime<Utc>>,
    pub owner_country: Option<String>,
    pub data_source: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// TLE entry for building ingestion batches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TleInput {
    pub norad_id: u32,
    pub line1: String,
    pub line2: String,
    pub epoch: DateTime<Utc>,
    pub source: String,
}

/// State vector entry (for export completeness)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateVectorInput {
    pub norad_id: u32,
    pub epoch: DateTime<Utc>,
    pub position_km: [f64; 3],
    pub velocity_kms: [f64; 3],
    pub covariance: Option<[f64; 21]>,
    pub reference_frame: String,
    pub quality: f64,
    pub source: String,
}

/// Input matching the `submit_observation` zome function in observations coordinator.
/// Fields match the observations integrity `Observation` entry type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitObservationCall {
    pub norad_id: Option<u32>,
    pub observation_time: SpaceTimestamp,
    pub observer_location: Option<GroundLocation>,
    pub observation_type: String, // "Radar" — serialized ObservationType variant
    pub measurement: Measurement,
    pub quality: u8,
    pub sensor_id: String,
}

/// SpaceTimestamp matching shared zome type (microseconds since epoch)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceTimestamp {
    pub micros: i64,
}

impl SpaceTimestamp {
    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self {
            micros: dt.timestamp_micros(),
        }
    }
}

/// GroundLocation matching shared zome type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundLocation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f64,
    pub name: Option<String>,
}

/// Measurement enum matching observations integrity zome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Measurement {
    StateVector {
        position_km: [f64; 3],
        velocity_kms: Option<[f64; 3]>,
        covariance: Option<[f64; 21]>,
    },
    Range {
        range_km: f64,
        range_rate_kms: Option<f64>,
        range_sigma_km: Option<f64>,
    },
    AnglesOnly {
        ra_deg: f64,
        dec_deg: f64,
        ra_sigma_deg: Option<f64>,
        dec_sigma_deg: Option<f64>,
    },
    Photometric {
        magnitude: f64,
        magnitude_sigma: Option<f64>,
        filter: Option<String>,
    },
}

/// Result from writing a batch file
#[derive(Debug, Clone)]
pub struct ZomeCallResult {
    pub file_path: String,
    pub _zome_name: String,
    pub _fn_name: String,
}

/// Holochain client that writes JSON files for batch import
pub struct HolochainClient {
    config: HolochainConfig,
    initialized: bool,
}

impl HolochainClient {
    pub fn new(config: HolochainConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(HolochainConfig::default())
    }

    /// Initialize the batch directory
    pub fn init(&mut self) -> Result<()> {
        std::fs::create_dir_all(&self.config.batch_dir)
            .context("Failed to create batch directory")?;
        self.initialized = true;
        println!("Batch directory ready: {}", self.config.batch_dir.display());
        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn batch_dir(&self) -> &Path {
        &self.config.batch_dir
    }

    /// Write a register_object zome call as JSON
    pub fn create_orbital_object(&self, input: OrbitalObjectInput) -> Result<ZomeCallResult> {
        if !self.initialized {
            anyhow::bail!("Client not initialized — call init() first");
        }

        let call = RegisterObjectCall {
            norad_id: input.norad_id,
            intl_designator: format!("CelesTrak-{}", input.norad_id),
            name: input.name,
            object_type: input.object_type,
            country: input.owner_country,
            launch_date: input.launch_date.map(|d| d.timestamp_micros()),
            status: Some("Unknown".to_string()),
        };

        let filename = format!("register_object_{}.json", input.norad_id);
        let path = self.config.batch_dir.join(&filename);
        let json = serde_json::to_string_pretty(&call)?;
        std::fs::write(&path, &json)?;

        Ok(ZomeCallResult {
            file_path: path.display().to_string(),
            _zome_name: "orbital_objects".to_string(),
            _fn_name: "register_object".to_string(),
        })
    }

    /// Write a submit_tle zome call as JSON
    pub fn create_tle(&self, input: TleInput) -> Result<ZomeCallResult> {
        if !self.initialized {
            anyhow::bail!("Client not initialized — call init() first");
        }

        let call = SubmitTleCall {
            norad_id: input.norad_id,
            line1: input.line1,
            line2: input.line2,
            source: Some(input.source),
            quality: Some(90),
        };

        let filename = format!("submit_tle_{}.json", input.norad_id);
        let path = self.config.batch_dir.join(&filename);
        let json = serde_json::to_string_pretty(&call)?;
        std::fs::write(&path, &json)?;

        Ok(ZomeCallResult {
            file_path: path.display().to_string(),
            _zome_name: "orbital_objects".to_string(),
            _fn_name: "submit_tle".to_string(),
        })
    }

    /// Write a state vector (stored alongside for reference, not directly a zome call)
    pub fn create_state_vector(&self, input: StateVectorInput) -> Result<ZomeCallResult> {
        if !self.initialized {
            anyhow::bail!("Client not initialized — call init() first");
        }

        let filename = format!("state_vector_{}.json", input.norad_id);
        let path = self.config.batch_dir.join(&filename);
        let json = serde_json::to_string_pretty(&input)?;
        std::fs::write(&path, &json)?;

        Ok(ZomeCallResult {
            file_path: path.display().to_string(),
            _zome_name: "orbital_objects".to_string(),
            _fn_name: "state_vector_reference".to_string(),
        })
    }

    /// Write a submit_observation zome call as JSON (observations coordinator)
    pub fn create_observation(&self, call: SubmitObservationCall) -> Result<ZomeCallResult> {
        if !self.initialized {
            anyhow::bail!("Client not initialized — call init() first");
        }

        let norad_suffix = call.norad_id.unwrap_or(0);
        let filename = format!(
            "submit_observation_{}_{}.json",
            norad_suffix, call.observation_time.micros
        );
        let path = self.config.batch_dir.join(&filename);
        let json = serde_json::to_string_pretty(&call)?;
        std::fs::write(&path, &json)?;

        Ok(ZomeCallResult {
            file_path: path.display().to_string(),
            _zome_name: "observations_coordinator".to_string(),
            _fn_name: "submit_observation".to_string(),
        })
    }

    /// Batch create orbital objects
    pub fn batch_create_objects(
        &self,
        objects: Vec<OrbitalObjectInput>,
    ) -> Result<Vec<ZomeCallResult>> {
        let mut results = Vec::new();
        for obj in objects {
            results.push(self.create_orbital_object(obj)?);
        }
        Ok(results)
    }
}

/// Builder for batch ingestion
pub struct IngestionBatch {
    objects: Vec<OrbitalObjectInput>,
    tles: Vec<TleInput>,
    state_vectors: Vec<StateVectorInput>,
    observations: Vec<SubmitObservationCall>,
}

impl IngestionBatch {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            tles: Vec::new(),
            state_vectors: Vec::new(),
            observations: Vec::new(),
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

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn add_observation(mut self, obs: SubmitObservationCall) -> Self {
        self.observations.push(obs);
        self
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

    /// Write all data as JSON files for batch import
    pub fn write_batch(&self, client: &HolochainClient) -> Result<IngestionReport> {
        let mut report = IngestionReport::new();

        for obj in &self.objects {
            match client.create_orbital_object(obj.clone()) {
                Ok(result) => {
                    report.objects_created += 1;
                    report.files_written.push(result.file_path);
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Failed to write object {}: {}", obj.norad_id, e));
                }
            }
        }

        for tle in &self.tles {
            match client.create_tle(tle.clone()) {
                Ok(result) => {
                    report.tles_created += 1;
                    report.files_written.push(result.file_path);
                }
                Err(e) => {
                    report
                        .errors
                        .push(format!("Failed to write TLE {}: {}", tle.norad_id, e));
                }
            }
        }

        for sv in &self.state_vectors {
            match client.create_state_vector(sv.clone()) {
                Ok(result) => {
                    report.state_vectors_created += 1;
                    report.files_written.push(result.file_path);
                }
                Err(e) => {
                    report.errors.push(format!(
                        "Failed to write state vector {}: {}",
                        sv.norad_id, e
                    ));
                }
            }
        }

        for obs in &self.observations {
            match client.create_observation(obs.clone()) {
                Ok(result) => {
                    report.observations_created += 1;
                    report.files_written.push(result.file_path);
                }
                Err(e) => {
                    let norad = obs.norad_id.unwrap_or(0);
                    report
                        .errors
                        .push(format!("Failed to write observation {}: {}", norad, e));
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

/// Report from batch file writing
#[derive(Debug, Default)]
pub struct IngestionReport {
    pub objects_created: usize,
    pub tles_created: usize,
    pub state_vectors_created: usize,
    pub observations_created: usize,
    pub files_written: Vec<String>,
    pub errors: Vec<String>,
}

impl IngestionReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_created(&self) -> usize {
        self.objects_created
            + self.tles_created
            + self.state_vectors_created
            + self.observations_created
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn print_summary(&self) {
        println!("\n=== Ingestion Report ===");
        println!("Orbital Objects: {}", self.objects_created);
        println!("TLEs: {}", self.tles_created);
        println!("State Vectors: {}", self.state_vectors_created);
        println!("Observations: {}", self.observations_created);
        println!("Total Files Written: {}", self.files_written.len());

        if self.has_errors() {
            println!("\nErrors ({}):", self.errors.len());
            for err in &self.errors {
                println!("  - {}", err);
            }
        } else {
            println!("\nNo errors during batch export.");
        }

        if !self.files_written.is_empty() {
            println!("\nTo import into Holochain, run:");
            println!(
                "  ./tools/import-batch.sh {}",
                self.files_written
                    .first()
                    .map(|f| Path::new(f)
                        .parent()
                        .unwrap_or(Path::new("."))
                        .display()
                        .to_string())
                    .unwrap_or_default()
            );
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
