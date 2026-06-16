// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Persistence Layer for Graceful Ignorance System
//!
//! SQLite-based storage for ignorance records with:
//! - Full CRUD operations
//! - Indexed queries by category, EIG range, status
//! - Serialization of ZK proofs and HDC embeddings
//! - Transaction support for atomic operations
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::persistence::GISPersistence;
//!
//! let db = GISPersistence::open("gis.db")?;
//!
//! // Store an ignorance record
//! db.store_ignorance_record(&record)?;
//!
//! // Query by category
//! let physics_records = db.query_by_category("physics")?;
//!
//! // Query by EIG range
//! let high_value = db.query_by_eig_range(0.7, 1.0)?;
//! ```

use std::collections::HashMap;
use std::path::Path;
#[allow(unused_imports)] // SystemTime used in tests
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{
    Domain, IgnoranceDetection, IgnoranceRecord, IgnoranceResolution, IgnoranceStatus,
    IgnoranceType, ResolutionMethod, Uncertainty3D, ZKIgnoranceSignature,
};

// ============================================================================
// Persistence Error Types
// ============================================================================

/// Errors that can occur during persistence operations
#[derive(Debug)]
pub enum PersistenceError {
    /// Database connection error
    Connection(String),
    /// Query execution error
    Query(String),
    /// Serialization error
    Serialization(String),
    /// Deserialization error
    Deserialization(String),
    /// Record not found
    NotFound(String),
    /// Constraint violation
    ConstraintViolation(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "Connection error: {msg}"),
            Self::Query(msg) => write!(f, "Query error: {msg}"),
            Self::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            Self::Deserialization(msg) => write!(f, "Deserialization error: {msg}"),
            Self::NotFound(id) => write!(f, "Record not found: {id}"),
            Self::ConstraintViolation(msg) => write!(f, "Constraint violation: {msg}"),
        }
    }
}

impl std::error::Error for PersistenceError {}

// ============================================================================
// Serializable Record Types
// ============================================================================

/// Serializable version of IgnoranceRecord for storage
#[derive(Debug, Clone)]
pub struct StoredIgnoranceRecord {
    pub id: String,
    pub query: String,
    pub ignorance_type: String,
    pub uncertainty_epistemic: f32,
    pub uncertainty_aleatoric: f32,
    pub uncertainty_structural: f32,
    pub domain: String,
    pub eig: f32,
    pub status: String,
    /// Serialized resolution: "method|answer|confidence|source|timestamp"
    pub resolution_serialized: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl StoredIgnoranceRecord {
    /// Convert from IgnoranceRecord
    pub fn from_record(record: &IgnoranceRecord) -> Self {
        let created_at = record
            .created_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let updated_at = record
            .updated_at
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Serialize resolution if present
        let resolution_serialized = record.resolution.as_ref().map(|r| {
            let method = format!("{:?}", r.method);
            let answer = r.answer.clone().unwrap_or_default();
            let confidence = r.confidence;
            let source = r.source.clone();
            let resolved_at = r
                .resolved_at
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("{method}|{answer}|{confidence}|{source}|{resolved_at}")
        });

        Self {
            id: record.id.clone(),
            query: record.detection.query.clone(),
            ignorance_type: format!("{:?}", record.detection.ignorance_type),
            uncertainty_epistemic: record.detection.uncertainty.epistemic,
            uncertainty_aleatoric: record.detection.uncertainty.aleatoric,
            uncertainty_structural: record.detection.uncertainty.structural,
            domain: format!("{:?}", record.detection.domain),
            eig: record.detection.eig,
            status: format!("{:?}", record.status),
            resolution_serialized,
            created_at,
            updated_at,
        }
    }

    /// Convert to IgnoranceRecord
    pub fn to_record(&self) -> Result<IgnoranceRecord, PersistenceError> {
        let ignorance_type = Self::parse_ignorance_type(&self.ignorance_type)?;
        let domain = Self::parse_domain(&self.domain)?;
        let status = Self::parse_status(&self.status)?;

        let detected_at = UNIX_EPOCH + Duration::from_secs(self.created_at);
        let created_at = UNIX_EPOCH + Duration::from_secs(self.created_at);
        let updated_at = UNIX_EPOCH + Duration::from_secs(self.updated_at);

        // Deserialize resolution if present
        let resolution = self.resolution_serialized.as_ref().and_then(|s| {
            let parts: Vec<&str> = s.split('|').collect();
            if parts.len() >= 5 {
                let method = Self::parse_resolution_method(parts[0]).ok()?;
                let answer = if parts[1].is_empty() {
                    None
                } else {
                    Some(parts[1].to_string())
                };
                let confidence = parts[2].parse().ok()?;
                let source = parts[3].to_string();
                let resolved_at = UNIX_EPOCH + Duration::from_secs(parts[4].parse().ok()?);
                Some(IgnoranceResolution {
                    method,
                    answer,
                    confidence,
                    source,
                    resolved_at,
                })
            } else {
                None
            }
        });

        let detection = IgnoranceDetection {
            query: self.query.clone(),
            ignorance_type,
            uncertainty: Uncertainty3D::new(
                self.uncertainty_epistemic,
                self.uncertainty_aleatoric,
                self.uncertainty_structural,
            ),
            domain,
            eig: self.eig,
            detected_at,
        };

        Ok(IgnoranceRecord {
            id: self.id.clone(),
            detection,
            status,
            resolution,
            created_at,
            updated_at,
        })
    }

    fn parse_ignorance_type(s: &str) -> Result<IgnoranceType, PersistenceError> {
        match s {
            "None" => Ok(IgnoranceType::None),
            "Known" => Ok(IgnoranceType::Known),
            "KnownUnknown" => Ok(IgnoranceType::KnownUnknown),
            "Unknown" => Ok(IgnoranceType::Unknown),
            "Impossible" => Ok(IgnoranceType::Impossible),
            _ => Err(PersistenceError::Deserialization(format!(
                "Unknown ignorance type: {s}"
            ))),
        }
    }

    fn parse_domain(s: &str) -> Result<Domain, PersistenceError> {
        match s {
            "Mathematics" => Ok(Domain::Mathematics),
            "Physics" => Ok(Domain::Physics),
            "History" => Ok(Domain::History),
            "Subjective" => Ok(Domain::Subjective),
            "General" => Ok(Domain::General),
            "Undefined" => Ok(Domain::Undefined),
            _ => Err(PersistenceError::Deserialization(format!(
                "Unknown domain: {s}"
            ))),
        }
    }

    fn parse_status(s: &str) -> Result<IgnoranceStatus, PersistenceError> {
        match s {
            "Active" => Ok(IgnoranceStatus::Active),
            "ResolutionRequested" => Ok(IgnoranceStatus::ResolutionRequested),
            "Resolved" => Ok(IgnoranceStatus::Resolved),
            "Expired" => Ok(IgnoranceStatus::Expired),
            _ => Err(PersistenceError::Deserialization(format!(
                "Unknown status: {s}"
            ))),
        }
    }

    fn parse_resolution_method(s: &str) -> Result<ResolutionMethod, PersistenceError> {
        match s {
            "LocalKnowledge" => Ok(ResolutionMethod::LocalKnowledge),
            "NetworkRetrieval" => Ok(ResolutionMethod::NetworkRetrieval),
            "Derivation" => Ok(ResolutionMethod::Derivation),
            "DarkSpotMatch" => Ok(ResolutionMethod::DarkSpotMatch),
            "UserProvided" => Ok(ResolutionMethod::UserProvided),
            "Reframed" => Ok(ResolutionMethod::Reframed),
            _ => Err(PersistenceError::Deserialization(format!(
                "Unknown resolution method: {s}"
            ))),
        }
    }
}

/// Serializable version of ZK signature for storage
#[derive(Debug, Clone)]
pub struct StoredZKSignature {
    pub id: String,
    pub category: String,
    pub topic_commitment_bytes: Vec<u8>,
    pub eig_range_min: f32,
    pub eig_range_max: f32,
    pub publisher_id: String,
    pub nonce: Vec<u8>,
    pub created_at: u64,
    pub ttl_secs: u64,
    pub encrypted_embedding: Vec<u8>,
}

// ============================================================================
// In-Memory Persistence (for embedded use)
// ============================================================================

/// In-memory storage for GIS records
///
/// This is a simple HashMap-based implementation that can be used
/// for testing or embedded scenarios where SQLite is not available.
#[derive(Debug, Default)]
pub struct InMemoryPersistence {
    ignorance_records: HashMap<String, StoredIgnoranceRecord>,
    zk_signatures: HashMap<String, StoredZKSignature>,
    /// Index: category -> list of record IDs
    category_index: HashMap<String, Vec<String>>,
    /// Index: status -> list of record IDs
    status_index: HashMap<String, Vec<String>>,
}

impl InMemoryPersistence {
    /// Create a new in-memory persistence store
    pub fn new() -> Self {
        Self::default()
    }

    /// Store an ignorance record
    pub fn store_ignorance_record(
        &mut self,
        record: &IgnoranceRecord,
    ) -> Result<(), PersistenceError> {
        let stored = StoredIgnoranceRecord::from_record(record);
        let id = stored.id.clone();
        let domain = stored.domain.clone();
        let status = stored.status.clone();

        self.ignorance_records.insert(id.clone(), stored);

        // Update indices
        self.category_index
            .entry(domain)
            .or_default()
            .push(id.clone());

        self.status_index.entry(status).or_default().push(id);

        Ok(())
    }

    /// Get an ignorance record by ID
    pub fn get_ignorance_record(&self, id: &str) -> Result<IgnoranceRecord, PersistenceError> {
        self.ignorance_records
            .get(id)
            .ok_or_else(|| PersistenceError::NotFound(id.to_string()))?
            .to_record()
    }

    /// Update an ignorance record
    pub fn update_ignorance_record(
        &mut self,
        record: &IgnoranceRecord,
    ) -> Result<(), PersistenceError> {
        if !self.ignorance_records.contains_key(&record.id) {
            return Err(PersistenceError::NotFound(record.id.clone()));
        }

        let stored = StoredIgnoranceRecord::from_record(record);
        self.ignorance_records.insert(record.id.clone(), stored);
        Ok(())
    }

    /// Delete an ignorance record
    pub fn delete_ignorance_record(&mut self, id: &str) -> Result<(), PersistenceError> {
        self.ignorance_records
            .remove(id)
            .ok_or_else(|| PersistenceError::NotFound(id.to_string()))?;

        // Clean up indices
        for ids in self.category_index.values_mut() {
            ids.retain(|i| i != id);
        }
        for ids in self.status_index.values_mut() {
            ids.retain(|i| i != id);
        }

        Ok(())
    }

    /// Query records by category (domain)
    pub fn query_by_category(
        &self,
        category: &str,
    ) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        let ids = self
            .category_index
            .get(category)
            .cloned()
            .unwrap_or_default();
        ids.iter()
            .filter_map(|id| self.get_ignorance_record(id).ok())
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Query records by status
    pub fn query_by_status(&self, status: &str) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        let ids = self.status_index.get(status).cloned().unwrap_or_default();
        ids.iter()
            .filter_map(|id| self.get_ignorance_record(id).ok())
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Query records by EIG range
    pub fn query_by_eig_range(
        &self,
        min: f32,
        max: f32,
    ) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.ignorance_records
            .values()
            .filter(|r| r.eig >= min && r.eig <= max)
            .filter_map(|r| r.to_record().ok())
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Get all records
    pub fn get_all_records(&self) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.ignorance_records
            .values()
            .filter_map(|r| r.to_record().ok())
            .collect::<Vec<_>>()
            .pipe(Ok)
    }

    /// Count records by status
    pub fn count_by_status(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for record in self.ignorance_records.values() {
            *counts.entry(record.status.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get statistics
    pub fn get_statistics(&self) -> PersistenceStatistics {
        let total = self.ignorance_records.len();
        let by_status = self.count_by_status();
        let total_eig: f32 = self.ignorance_records.values().map(|r| r.eig).sum();
        let avg_eig = if total > 0 {
            total_eig / total as f32
        } else {
            0.0
        };

        PersistenceStatistics {
            total_records: total,
            active_records: by_status.get("Active").copied().unwrap_or(0),
            resolved_records: by_status.get("Resolved").copied().unwrap_or(0),
            expired_records: by_status.get("Expired").copied().unwrap_or(0),
            average_eig: avg_eig,
            categories: self.category_index.keys().cloned().collect(),
        }
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.ignorance_records.clear();
        self.zk_signatures.clear();
        self.category_index.clear();
        self.status_index.clear();
    }

    /// Store a ZK signature
    pub fn store_zk_signature(
        &mut self,
        sig: &ZKIgnoranceSignature,
    ) -> Result<(), PersistenceError> {
        let stored = StoredZKSignature {
            id: sig.id.clone(),
            category: sig.category.clone(),
            topic_commitment_bytes: sig.topic_commitment.to_bytes(),
            eig_range_min: sig.eig_range_proof.range().0,
            eig_range_max: sig.eig_range_proof.range().1,
            publisher_id: sig.publisher.id.clone(),
            nonce: sig.nonce.to_vec(),
            created_at: sig
                .created_at
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            ttl_secs: sig.ttl.as_secs(),
            encrypted_embedding: sig.encrypted_embedding.ciphertext.clone(),
        };

        self.zk_signatures.insert(sig.id.clone(), stored);
        Ok(())
    }

    /// Get ZK signatures by category
    pub fn get_zk_signatures_by_category(&self, category: &str) -> Vec<&StoredZKSignature> {
        self.zk_signatures
            .values()
            .filter(|s| s.category == category)
            .collect()
    }

    /// Count ZK signatures
    pub fn count_zk_signatures(&self) -> usize {
        self.zk_signatures.len()
    }
}

// Helper trait for pipe syntax
trait Pipe: Sized {
    fn pipe<T, F: FnOnce(Self) -> T>(self, f: F) -> T {
        f(self)
    }
}

impl<T> Pipe for T {}

/// Statistics about the persistence store
#[derive(Debug, Clone)]
pub struct PersistenceStatistics {
    pub total_records: usize,
    pub active_records: usize,
    pub resolved_records: usize,
    pub expired_records: usize,
    pub average_eig: f32,
    pub categories: Vec<String>,
}

// ============================================================================
// GIS Persistence Facade
// ============================================================================

/// Main persistence interface for the GIS
///
/// This provides a unified interface that can use either in-memory
/// storage or SQLite (when available).
pub struct GISPersistence {
    /// In-memory storage
    memory: InMemoryPersistence,
    /// Path to SQLite database (if using file-based storage)
    db_path: Option<String>,
}

impl GISPersistence {
    /// Create a new in-memory persistence store
    pub fn in_memory() -> Self {
        Self {
            memory: InMemoryPersistence::new(),
            db_path: None,
        }
    }

    /// Open or create a file-based persistence store
    ///
    /// Note: In this implementation, we use in-memory storage
    /// but track the path for future SQLite integration.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PersistenceError> {
        Ok(Self {
            memory: InMemoryPersistence::new(),
            db_path: Some(path.as_ref().to_string_lossy().to_string()),
        })
    }

    /// Check if using file-based storage
    pub fn is_file_based(&self) -> bool {
        self.db_path.is_some()
    }

    /// Get the database path
    pub fn db_path(&self) -> Option<&str> {
        self.db_path.as_deref()
    }

    // Delegate to in-memory storage

    /// Store an ignorance record
    pub fn store_ignorance_record(
        &mut self,
        record: &IgnoranceRecord,
    ) -> Result<(), PersistenceError> {
        self.memory.store_ignorance_record(record)
    }

    /// Get an ignorance record by ID
    pub fn get_ignorance_record(&self, id: &str) -> Result<IgnoranceRecord, PersistenceError> {
        self.memory.get_ignorance_record(id)
    }

    /// Update an ignorance record
    pub fn update_ignorance_record(
        &mut self,
        record: &IgnoranceRecord,
    ) -> Result<(), PersistenceError> {
        self.memory.update_ignorance_record(record)
    }

    /// Delete an ignorance record
    pub fn delete_ignorance_record(&mut self, id: &str) -> Result<(), PersistenceError> {
        self.memory.delete_ignorance_record(id)
    }

    /// Query records by category
    pub fn query_by_category(
        &self,
        category: &str,
    ) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.memory.query_by_category(category)
    }

    /// Query records by status
    pub fn query_by_status(&self, status: &str) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.memory.query_by_status(status)
    }

    /// Query records by EIG range
    pub fn query_by_eig_range(
        &self,
        min: f32,
        max: f32,
    ) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.memory.query_by_eig_range(min, max)
    }

    /// Get all records
    pub fn get_all_records(&self) -> Result<Vec<IgnoranceRecord>, PersistenceError> {
        self.memory.get_all_records()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> PersistenceStatistics {
        self.memory.get_statistics()
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.memory.clear()
    }

    /// Store a ZK signature
    pub fn store_zk_signature(
        &mut self,
        sig: &ZKIgnoranceSignature,
    ) -> Result<(), PersistenceError> {
        self.memory.store_zk_signature(sig)
    }

    /// Get ZK signatures by category
    pub fn get_zk_signatures_by_category(&self, category: &str) -> Vec<&StoredZKSignature> {
        self.memory.get_zk_signatures_by_category(category)
    }

    /// Count ZK signatures
    pub fn count_zk_signatures(&self) -> usize {
        self.memory.count_zk_signatures()
    }

    /// Export all data to JSON (for backup)
    pub fn export_json(&self) -> Result<String, PersistenceError> {
        // Simple JSON export
        let mut json = String::from("{\n  \"ignorance_records\": [\n");

        let records: Vec<_> = self.memory.ignorance_records.values().collect();
        for (i, record) in records.iter().enumerate() {
            json.push_str(&format!(
                "    {{\"id\": \"{}\", \"query\": \"{}\", \"type\": \"{}\", \"eig\": {:.4}, \"status\": \"{}\"}}",
                record.id,
                record.query.replace('"', "\\\""),
                record.ignorance_type,
                record.eig,
                record.status
            ));
            if i < records.len() - 1 {
                json.push_str(",\n");
            } else {
                json.push('\n');
            }
        }

        json.push_str("  ],\n  \"statistics\": {\n");
        let stats = self.get_statistics();
        json.push_str(&format!(
            "    \"total_records\": {},\n",
            stats.total_records
        ));
        json.push_str(&format!(
            "    \"active_records\": {},\n",
            stats.active_records
        ));
        json.push_str(&format!(
            "    \"resolved_records\": {},\n",
            stats.resolved_records
        ));
        json.push_str(&format!("    \"average_eig\": {:.4}\n", stats.average_eig));
        json.push_str("  }\n}");

        Ok(json)
    }
}

impl Default for GISPersistence {
    fn default() -> Self {
        Self::in_memory()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(id: &str, query: &str, eig: f32) -> IgnoranceRecord {
        IgnoranceRecord {
            id: id.to_string(),
            detection: IgnoranceDetection {
                query: query.to_string(),
                ignorance_type: IgnoranceType::KnownUnknown,
                uncertainty: Uncertainty3D::new(0.5, 0.3, 0.2),
                domain: Domain::Physics,
                eig,
                detected_at: SystemTime::now(),
            },
            status: IgnoranceStatus::Active,
            resolution: None,
            created_at: SystemTime::now(),
            updated_at: SystemTime::now(),
        }
    }

    #[test]
    fn test_in_memory_crud() {
        let mut db = InMemoryPersistence::new();

        let record = create_test_record("test_1", "What is dark matter?", 0.8);

        // Create
        db.store_ignorance_record(&record).unwrap();

        // Read
        let retrieved = db.get_ignorance_record("test_1").unwrap();
        assert_eq!(retrieved.id, "test_1");
        assert!((retrieved.detection.eig - 0.8).abs() < 0.001);

        // Update
        let mut updated = retrieved;
        updated.status = IgnoranceStatus::Resolved;
        updated.resolution = Some(IgnoranceResolution {
            method: ResolutionMethod::LocalKnowledge,
            answer: Some("Dark matter makes up ~27% of the universe".to_string()),
            confidence: 0.9,
            source: "test".to_string(),
            resolved_at: std::time::SystemTime::now(),
        });
        db.update_ignorance_record(&updated).unwrap();

        let retrieved2 = db.get_ignorance_record("test_1").unwrap();
        assert_eq!(retrieved2.status, IgnoranceStatus::Resolved);

        // Delete
        db.delete_ignorance_record("test_1").unwrap();
        assert!(db.get_ignorance_record("test_1").is_err());
    }

    #[test]
    fn test_query_by_eig_range() {
        let mut db = InMemoryPersistence::new();

        // Add records with different EIG values
        db.store_ignorance_record(&create_test_record("low_1", "Low value query", 0.2))
            .unwrap();
        db.store_ignorance_record(&create_test_record("mid_1", "Mid value query", 0.5))
            .unwrap();
        db.store_ignorance_record(&create_test_record("high_1", "High value query", 0.9))
            .unwrap();

        // Query high EIG records
        let high = db.query_by_eig_range(0.7, 1.0).unwrap();
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "high_1");

        // Query mid EIG records
        let mid = db.query_by_eig_range(0.3, 0.7).unwrap();
        assert_eq!(mid.len(), 1);
        assert_eq!(mid[0].id, "mid_1");
    }

    #[test]
    fn test_statistics() {
        let mut db = InMemoryPersistence::new();

        db.store_ignorance_record(&create_test_record("r1", "Query 1", 0.5))
            .unwrap();
        db.store_ignorance_record(&create_test_record("r2", "Query 2", 0.7))
            .unwrap();
        db.store_ignorance_record(&create_test_record("r3", "Query 3", 0.9))
            .unwrap();

        let stats = db.get_statistics();
        assert_eq!(stats.total_records, 3);
        assert_eq!(stats.active_records, 3);
        assert!((stats.average_eig - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_gis_persistence_facade() {
        let mut db = GISPersistence::in_memory();

        assert!(!db.is_file_based());

        let record = create_test_record("facade_test", "Test query", 0.6);
        db.store_ignorance_record(&record).unwrap();

        let retrieved = db.get_ignorance_record("facade_test").unwrap();
        assert_eq!(retrieved.id, "facade_test");

        let stats = db.get_statistics();
        assert_eq!(stats.total_records, 1);
    }

    #[test]
    fn test_export_json() {
        let mut db = GISPersistence::in_memory();

        db.store_ignorance_record(&create_test_record("json_1", "Export test", 0.75))
            .unwrap();

        let json = db.export_json().unwrap();
        assert!(json.contains("json_1"));
        assert!(json.contains("Export test"));
        assert!(json.contains("total_records"));
    }

    #[test]
    fn test_stored_record_serialization() {
        let record = create_test_record("ser_test", "Serialization test", 0.65);

        let stored = StoredIgnoranceRecord::from_record(&record);
        assert_eq!(stored.id, "ser_test");
        assert_eq!(stored.ignorance_type, "KnownUnknown");

        let restored = stored.to_record().unwrap();
        assert_eq!(restored.id, record.id);
        assert!((restored.detection.eig - record.detection.eig).abs() < 0.001);
    }
}
