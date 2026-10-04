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
    Domain, EpistemicFrame, EpistemicFrameImpact, EpistemicFrameRevision, IgnoranceDetection, IgnoranceRecord,
    IgnoranceResolution, IgnoranceStatus, IgnoranceType, ResolutionMethod, Uncertainty3D,
    ZKIgnoranceSignature,
};

// ============================================================================
// Frame revision serialization
// ============================================================================

const FRAME_REVISION_FORMAT_VERSION: &str = "v2";

fn push_len_prefixed(output: &mut String, value: &str) {
    use std::fmt::Write as _;
    write!(output, "{}:", value.len()).expect("writing to String cannot fail");
    output.push_str(value);
}

fn take_len_prefixed(input: &str, cursor: &mut usize) -> Result<String, PersistenceError> {
    let bytes = input.as_bytes();
    let colon = bytes
        .get(*cursor..)
        .and_then(|tail| tail.iter().position(|byte| *byte == b':'))
        .map(|offset| *cursor + offset)
        .ok_or_else(|| PersistenceError::Deserialization(
            "frame revision is missing a length delimiter".into(),
        ))?;

    let len = std::str::from_utf8(&bytes[*cursor..colon])
        .map_err(|_| PersistenceError::Deserialization(
            "frame revision length is not valid UTF-8".into(),
        ))?
        .parse::<usize>()
        .map_err(|_| PersistenceError::Deserialization(
            "frame revision length is not numeric".into(),
        ))?;

    let start = colon + 1;
    let end = start
        .checked_add(len)
        .ok_or_else(|| PersistenceError::Deserialization(
            "frame revision length overflow".into(),
        ))?;
    if end > bytes.len() {
        return Err(PersistenceError::Deserialization(
            "frame revision length exceeds payload".into(),
        ));
    }

    let value = std::str::from_utf8(&bytes[start..end])
        .map_err(|_| PersistenceError::Deserialization(
            "frame revision payload is not valid UTF-8".into(),
        ))?
        .to_owned();
    *cursor = end;
    Ok(value)
}

fn serialize_frame_revision(revision: &EpistemicFrameRevision) -> String {
    let mut output = String::from(FRAME_REVISION_FORMAT_VERSION);
    output.push('|');
    for value in [
        revision.prior_frame.as_str(),
        revision.revised_frame.as_str(),
        revision.trigger.as_str(),
        revision.newly_represented.as_deref().unwrap_or(""),
        revision.scope_change.as_str(),
    ] {
        push_len_prefixed(&mut output, value);
    }

    push_len_prefixed(&mut output, &revision.affected_conclusions.len().to_string());
    for conclusion in &revision.affected_conclusions {
        push_len_prefixed(&mut output, conclusion);
    }

    for value in [
        revision.impact.evidence_boundary,
        revision.impact.ontology,
        revision.impact.causal_model,
        revision.impact.exclusions,
        revision.impact.blind_spots,
    ] {
        push_len_prefixed(&mut output, if value { "1" } else { "0" });
    }
    output
}

fn parse_frame_revision_v2(encoded: &str) -> Result<EpistemicFrameRevision, PersistenceError> {
    let payload = encoded
        .strip_prefix("v2|")
        .ok_or_else(|| PersistenceError::Deserialization(
            "frame revision has an unsupported format version".into(),
        ))?;
    let mut cursor = 0usize;

    let prior_frame = take_len_prefixed(payload, &mut cursor)?;
    let revised_frame = take_len_prefixed(payload, &mut cursor)?;
    let trigger = take_len_prefixed(payload, &mut cursor)?;
    let newly_represented_raw = take_len_prefixed(payload, &mut cursor)?;
    let scope_change = take_len_prefixed(payload, &mut cursor)?;
    let affected_count = take_len_prefixed(payload, &mut cursor)?
        .parse::<usize>()
        .map_err(|_| PersistenceError::Deserialization(
            "frame revision affected-conclusion count is not numeric".into(),
        ))?;

    let mut affected_conclusions = Vec::with_capacity(affected_count);
    for _ in 0..affected_count {
        affected_conclusions.push(take_len_prefixed(payload, &mut cursor)?);
    }

    let mut flags = [false; 5];
    for flag in &mut flags {
        *flag = match take_len_prefixed(payload, &mut cursor)?.as_str() {
            "0" => false,
            "1" => true,
            _ => {
                return Err(PersistenceError::Deserialization(
                    "frame revision impact flag is not boolean".into(),
                ))
            }
        };
    }

    if cursor != payload.len() {
        return Err(PersistenceError::Deserialization(
            "frame revision contains trailing bytes".into(),
        ));
    }

    Ok(EpistemicFrameRevision {
        prior_frame,
        revised_frame,
        trigger,
        newly_represented: if newly_represented_raw.is_empty() {
            None
        } else {
            Some(newly_represented_raw)
        },
        scope_change,
        affected_conclusions,
        impact: EpistemicFrameImpact {
            evidence_boundary: flags[0],
            ontology: flags[1],
            causal_model: flags[2],
            exclusions: flags[3],
            blind_spots: flags[4],
        },
    })
}

fn deserialize_frame_revision(encoded: &str) -> Result<EpistemicFrameRevision, PersistenceError> {
    if encoded.starts_with("v2|") {
        return parse_frame_revision_v2(encoded);
    }

    // Legacy six-field format retained for existing persisted records written by
    // earlier revisions of this branch. Legacy data remains best-effort; new data
    // is always emitted in the lossless v2 format above.
    let parts: Vec<&str> = encoded.split(';').collect();
    if parts.len() < 6 {
        return Err(PersistenceError::Deserialization(
            "legacy frame revision has too few fields".into(),
        ));
    }

    let impact = parts.get(6)
        .and_then(|encoded| {
            let flags: Vec<&str> = encoded.split(',').collect();
            if flags.len() != 5 {
                return None;
            }
            Some(EpistemicFrameImpact {
                evidence_boundary: flags[0].parse().ok()?,
                ontology: flags[1].parse().ok()?,
                causal_model: flags[2].parse().ok()?,
                exclusions: flags[3].parse().ok()?,
                blind_spots: flags[4].parse().ok()?,
            })
        })
        .unwrap_or_else(EpistemicFrameImpact::broad);

    Ok(EpistemicFrameRevision {
        prior_frame: parts[0].to_string(),
        revised_frame: parts[1].to_string(),
        trigger: parts[2].to_string(),
        newly_represented: if parts[3].is_empty() {
            None
        } else {
            Some(parts[3].to_string())
        },
        scope_change: parts[4].to_string(),
        affected_conclusions: parts[5]
            .split(',')
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
            .collect(),
        impact,
    })
}

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
    /// Stable frame identity and revision used to qualify the detection.
    pub frame_id: String,
    pub frame_version: u32,
    pub frame_evidence_boundary: String,
    pub frame_ontology_id: String,
    pub frame_causal_model_id: String,
    pub frame_excluded_variables: Vec<String>,
    pub frame_known_blind_spots: Vec<String>,
    /// Append-only serialized frame revision lineage.
    pub frame_revisions_serialized: Vec<String>,
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
            frame_id: record.detection.frame.id.clone(),
            frame_version: record.detection.frame.version,
            frame_evidence_boundary: record.detection.frame.evidence_boundary.clone(),
            frame_ontology_id: record.detection.frame.ontology_id.clone(),
            frame_causal_model_id: record.detection.frame.causal_model_id.clone(),
            frame_excluded_variables: record.detection.frame.excluded_variables.clone(),
            frame_known_blind_spots: record.detection.frame.known_blind_spots.clone(),
            frame_revisions_serialized: record
                .frame_revisions
                .iter()
                .map(serialize_frame_revision)
                .collect(),
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
            frame: EpistemicFrame {
                id: if self.frame_id.is_empty() { "gis-default".to_string() } else { self.frame_id.clone() },
                version: if self.frame_version == 0 { 1 } else { self.frame_version },
                evidence_boundary: if self.frame_evidence_boundary.is_empty() { "local-query-context".to_string() } else { self.frame_evidence_boundary.clone() },
                ontology_id: if self.frame_ontology_id.is_empty() { "general-v1".to_string() } else { self.frame_ontology_id.clone() },
                causal_model_id: if self.frame_causal_model_id.is_empty() { "unspecified".to_string() } else { self.frame_causal_model_id.clone() },
                excluded_variables: self.frame_excluded_variables.clone(),
                known_blind_spots: if self.frame_known_blind_spots.is_empty() { vec!["unrepresented variables and categories".to_string()] } else { self.frame_known_blind_spots.clone() },
            },
        };

        let frame_revisions = self
            .frame_revisions_serialized
            .iter()
            .map(deserialize_frame_revision)
            .collect::<Result<Vec<_>, _>>()?;

        let mut expected_frame = detection.frame.identity();
        for revision in &frame_revisions {
            if !revision.follows_frame(&expected_frame) || !revision.changes_frame() {
                return Err(PersistenceError::Deserialization(
                    "frame revision lineage is discontinuous or contains a no-op".into(),
                ));
            }
            expected_frame = revision.revised_frame.clone();
        }

        Ok(IgnoranceRecord {
            id: self.id.clone(),
            detection,
            status,
            resolution,
            frame_revisions,
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

        self.remove_from_indexes(&id);
        self.ignorance_records.insert(id.clone(), stored);

        self.category_index
            .entry(domain)
            .or_default()
            .push(id.clone());
        self.status_index.entry(status).or_default().push(id);

        Ok(())
    }

    fn remove_from_indexes(&mut self, id: &str) {
        for ids in self.category_index.values_mut() {
            ids.retain(|indexed_id| indexed_id != id);
        }
        for ids in self.status_index.values_mut() {
            ids.retain(|indexed_id| indexed_id != id);
        }
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
        let id = record.id.clone();
        let domain = stored.domain.clone();
        let status = stored.status.clone();

        self.remove_from_indexes(&id);
        self.ignorance_records.insert(id.clone(), stored);
        self.category_index.entry(domain).or_default().push(id.clone());
        self.status_index.entry(status).or_default().push(id);
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
                frame: EpistemicFrame::default(),
            },
            status: IgnoranceStatus::Active,
            resolution: None,
            frame_revisions: Vec::new(),
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
    fn test_update_refreshes_category_and_status_indexes() {
        let mut db = InMemoryPersistence::new();
        let mut record = create_test_record("index_update", "Index", 0.5);

        db.store_ignorance_record(&record).unwrap();
        assert_eq!(db.query_by_category("General").unwrap().len(), 1);
        assert_eq!(db.query_by_status("Active").unwrap().len(), 1);

        record.detection.domain = Domain::Physics;
        record.status = IgnoranceStatus::Resolved;
        db.update_ignorance_record(&record).unwrap();

        assert!(db.query_by_category("General").unwrap().is_empty());
        assert_eq!(db.query_by_category("Physics").unwrap().len(), 1);
        assert!(db.query_by_status("Active").unwrap().is_empty());
        assert_eq!(db.query_by_status("Resolved").unwrap().len(), 1);

        db.store_ignorance_record(&record).unwrap();
        assert_eq!(db.query_by_category("Physics").unwrap().len(), 1);
        assert_eq!(db.query_by_status("Resolved").unwrap().len(), 1);
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
        let mut record = create_test_record("ser_test", "Serialization test", 0.65);
        record.detection.frame = EpistemicFrame {
            id: "social-system".to_string(),
            version: 3,
            evidence_boundary: "institutional-records".to_string(),
            ontology_id: "collective-agents-v2".to_string(),
            causal_model_id: "institutional-feedback-v4".to_string(),
            excluded_variables: vec!["informal-practices".to_string()],
            known_blind_spots: vec!["unobserved local norms".to_string()],
        };

        let stored = StoredIgnoranceRecord::from_record(&record);
        assert_eq!(stored.id, "ser_test");
        assert_eq!(stored.ignorance_type, "KnownUnknown");

        let restored = stored.to_record().unwrap();
        assert_eq!(restored.id, record.id);
        assert!((restored.detection.eig - record.detection.eig).abs() < 0.001);
        assert_eq!(restored.detection.frame.identity(), "social-system@3");
        assert_eq!(restored.detection.frame.ontology_id, "collective-agents-v2");
        assert_eq!(restored.detection.frame.causal_model_id, "institutional-feedback-v4");
        assert_eq!(restored.detection.frame.excluded_variables, vec!["informal-practices"]);
        assert_eq!(restored.detection.frame.known_blind_spots, vec!["unobserved local norms"]);
    }

    #[test]
    fn test_frame_revision_lineage_round_trip() {
        let mut record = create_test_record("lineage_test", "What changed?", 0.8);
        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame {
            id: prior.id.clone(),
            version: prior.version + 1,
            evidence_boundary: "institutional-records".to_string(),
            ontology_id: "collective-agents-v2".to_string(),
            causal_model_id: prior.causal_model_id.clone(),
            excluded_variables: vec!["informal-practices".to_string()],
            known_blind_spots: vec!["unobserved local norms".to_string()],
        };
        record.append_frame_revision(EpistemicFrameRevision::new(
            &prior,
            &revised,
            "new institutional evidence",
            Some("informal-practices".to_string()),
            "expanded scope to include informal practices",
            vec!["conclusion-17".to_string(), "conclusion-23".to_string()],
        ));

        let stored = StoredIgnoranceRecord::from_record(&record);
        let restored = stored.to_record().unwrap();

        assert_eq!(restored.frame_revisions, record.frame_revisions);
        assert_eq!(
            restored.latest_frame_revision().unwrap().affected_conclusions,
            vec!["conclusion-17", "conclusion-23"]
        );
        let impact = restored.latest_frame_revision().unwrap().impact;
        assert!(impact.evidence_boundary);
        assert!(impact.ontology);
        assert!(!impact.causal_model);
        assert!(impact.exclusions);
        assert!(impact.blind_spots);
    }

    #[test]
    fn test_frame_revision_serialization_is_lossless_for_delimiters() {
        let mut record = create_test_record("delimiter_lineage", "Delimited", 0.4);
        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame {
            id: prior.id.clone(),
            version: prior.version + 1,
            ..prior.clone()
        };
        record.append_frame_revision(EpistemicFrameRevision {
            prior_frame: prior.identity(),
            revised_frame: revised.identity(),
            trigger: "trigger;with;semicolons,commas".to_string(),
            newly_represented: Some("entity;with,delimiters".to_string()),
            scope_change: "scope;change,with:punctuation".to_string(),
            affected_conclusions: vec![
                "conclusion;one".to_string(),
                "conclusion,two".to_string(),
                "conclusion:three".to_string(),
            ],
            impact: EpistemicFrameImpact {
                evidence_boundary: true,
                ontology: false,
                causal_model: true,
                exclusions: false,
                blind_spots: true,
            },
        });

        let stored = StoredIgnoranceRecord::from_record(&record);
        assert!(stored.frame_revisions_serialized[0].starts_with("v2|"));

        let restored = stored.to_record().unwrap();
        assert_eq!(restored.frame_revisions, record.frame_revisions);
    }

    #[test]
    fn test_discontinuous_frame_revision_lineage_fails_closed() {
        let mut record = create_test_record("discontinuous_lineage", "Discontinuous", 0.4);
        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame { version: prior.version + 1, ..prior.clone() };

        record.frame_revisions = vec![
            EpistemicFrameRevision::new(
                &prior,
                &revised,
                "first revision",
                None,
                "first scope change",
                vec!["c1".to_string()],
            ),
            EpistemicFrameRevision {
                prior_frame: "foreign-frame@9".to_string(),
                revised_frame: "foreign-frame@10".to_string(),
                trigger: "spliced".to_string(),
                newly_represented: None,
                scope_change: "foreign jump".to_string(),
                affected_conclusions: vec!["c2".to_string()],
                impact: EpistemicFrameImpact::broad(),
            },
        ];

        let stored = StoredIgnoranceRecord::from_record(&record);
        let error = stored.to_record().unwrap_err();
        assert!(matches!(error, PersistenceError::Deserialization(_)));
    }

    #[test]
    fn test_corrupt_v2_frame_revision_fails_closed() {
        let mut record = create_test_record("corrupt_lineage", "Corrupt", 0.4);
        record.frame_revisions_serialized = vec!["v2|4:gis-default".to_string()];

        let error = record.to_record().unwrap_err();
        assert!(matches!(error, PersistenceError::Deserialization(_)));
    }

    #[test]
    fn test_legacy_frame_revision_without_impact_is_conservative() {
        let record = create_test_record("legacy_lineage", "Legacy", 0.4);
        let prior = record.detection.frame.clone();
        let revised = EpistemicFrame { version: 2, ..prior.clone() };
        let revision = EpistemicFrameRevision::new(
            &prior, &revised, "legacy", None, "version bump", vec!["c1".to_string()],
        );

        let mut stored = StoredIgnoranceRecord::from_record(&record);
        stored.frame_revisions_serialized = vec![format!(
            "{};{};{};;{};{}",
            revision.prior_frame,
            revision.revised_frame,
            revision.trigger,
            revision.scope_change,
            revision.affected_conclusions.join(",")
        )];

        let restored = stored.to_record().unwrap();
        assert_eq!(
            restored.latest_frame_revision().unwrap().impact,
            EpistemicFrameImpact::broad()
        );
    }
}
